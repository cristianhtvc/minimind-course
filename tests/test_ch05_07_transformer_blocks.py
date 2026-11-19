"""第 5-7 章测试：RoPE/YaRN、GQA 注意力与 KV Cache、SwiGLU FFN（逐项对齐 MiniMind）"""
import torch

from conftest import load_module
from model.model_minimind import (
    MiniMindConfig,
    precompute_freqs_cis as mm_freqs,
    apply_rotary_pos_emb as mm_apply_rope,
    repeat_kv as mm_repeat_kv,
    Attention as MMAttention,
    FeedForward as MMFeedForward,
)
from tinymind.config import TinyMindConfig


def test_ch06_rope_and_yarn_match_minimind():
    rope = load_module("rope")
    for scaling in (None, {"beta_fast": 32, "beta_slow": 1, "factor": 16,
                           "original_max_position_embeddings": 128, "attention_factor": 1.0, "type": "yarn"}):
        cos, sin = rope.precompute_freqs_cis(dim=32, end=512, rope_base=1e6, rope_scaling=scaling)
        ref_cos, ref_sin = mm_freqs(dim=32, end=512, rope_base=1e6, rope_scaling=scaling)
        assert torch.allclose(cos, ref_cos, atol=1e-6)
        assert torch.allclose(sin, ref_sin, atol=1e-6)

    q = torch.randn(2, 10, 4, 32)
    k = torch.randn(2, 10, 2, 32)
    q1, k1 = rope.apply_rotary_pos_emb(q, k, cos[:10], sin[:10])
    q2, k2 = mm_apply_rope(q, k, ref_cos[:10], ref_sin[:10])
    assert torch.allclose(q1, q2, atol=1e-6)
    assert torch.allclose(k1, k2, atol=1e-6)

    # 核心性质：旋转后的内积只与相对位置 (m - n) 有关
    v1, v2 = torch.randn(1, 1, 1, 32), torch.randn(1, 1, 1, 32)
    q_m, _ = rope.apply_rotary_pos_emb(v1, v1, cos[7:8], sin[7:8])
    _, k_n = rope.apply_rotary_pos_emb(v2, v2, cos[3:4], sin[3:4])
    q_m2, _ = rope.apply_rotary_pos_emb(v1, v1, cos[27:28], sin[27:28])
    _, k_n2 = rope.apply_rotary_pos_emb(v2, v2, cos[23:24], sin[23:24])
    # 注意无 YaRN 时才严格满足普通频率下的平移不变；这里用无 scaling 的表检验
    cos0, sin0 = rope.precompute_freqs_cis(dim=32, end=64)
    a, _ = rope.apply_rotary_pos_emb(v1, v1, cos0[10:11], sin0[10:11])
    _, b = rope.apply_rotary_pos_emb(v2, v2, cos0[4:5], sin0[4:5])
    a2, _ = rope.apply_rotary_pos_emb(v1, v1, cos0[30:31], sin0[30:31])
    _, b2 = rope.apply_rotary_pos_emb(v2, v2, cos0[24:25], sin0[24:25])
    assert torch.allclose((a * b).sum(), (a2 * b2).sum(), atol=1e-5)


def test_ch05_attention_and_kv_cache():
    attn_mod = load_module("attention")
    rope_mod = load_module("rope")
    x_kv = torch.randn(2, 5, 2, 16)
    assert torch.equal(attn_mod.repeat_kv(x_kv, 2), mm_repeat_kv(x_kv, 2))

    for flash in (False, True):
        cfg = TinyMindConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=2, flash_attn=flash)
        mm_cfg = MiniMindConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=2, flash_attn=flash)
        ours = attn_mod.Attention(cfg).eval()
        ref = MMAttention(mm_cfg).eval()
        ref.load_state_dict(ours.state_dict())

        x = torch.randn(2, 8, 64)
        cos, sin = rope_mod.precompute_freqs_cis(cfg.head_dim, 32)
        out1, kv1 = ours(x, (cos[:8], sin[:8]), use_cache=True)
        out2, _ = ref(x, (cos[:8], sin[:8]), use_cache=True)
        assert torch.allclose(out1, out2, atol=1e-5)

        # 验证 KV Cache 增量推理 == 一次性全序列推理
        out_full, _ = ours(x, (cos[:8], sin[:8]), use_cache=False)
        out_prefill, cache = ours(x[:, :5], (cos[:5], sin[:5]), use_cache=True)
        step_outs = [out_prefill]
        for t in range(5, 8):
            o, cache = ours(x[:, t:t + 1], (cos[t:t + 1], sin[t:t + 1]), past_key_value=cache, use_cache=True)
            step_outs.append(o)
        assert torch.allclose(out_full, torch.cat(step_outs, dim=1), atol=1e-5)


def test_ch07_swiglu_ffn():
    ffn_mod = load_module("ffn")
    cfg = TinyMindConfig(hidden_size=64)
    mm_cfg = MiniMindConfig(hidden_size=64)
    ours = ffn_mod.FeedForward(cfg).eval()
    ref = MMFeedForward(mm_cfg).eval()
    ref.load_state_dict(ours.state_dict())
    x = torch.randn(2, 6, 64)
    assert torch.allclose(ours(x), ref(x), atol=1e-6)
