"""第 8-10 章测试：完整模型（Dense + MoE）与 MiniMind 权重互载 & 数值等价、自回归生成"""
import torch

from conftest import load_module
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from tinymind.config import TinyMindConfig


def test_ch08_09_full_model_and_moe_match_minimind():
    model_mod = load_module("model")
    for use_moe in (False, True):
        cfg = TinyMindConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                             num_key_value_heads=2, vocab_size=128, use_moe=use_moe, num_experts=4)
        mm_cfg = MiniMindConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                                num_key_value_heads=2, vocab_size=128, use_moe=use_moe, num_experts=4)
        ours = model_mod.TinyMindForCausalLM(cfg).train()
        ref = MiniMindForCausalLM(mm_cfg).train()
        # state_dict 键集合必须完全一致（可直接加载 MiniMind 开源权重）
        assert set(ours.state_dict().keys()) == set(ref.state_dict().keys())
        ref.load_state_dict(ours.state_dict())

        # tie_word_embeddings 必须生效
        assert ours.model.embed_tokens.weight is ours.lm_head.weight

        input_ids = torch.randint(0, 128, (3, 12))
        labels = input_ids.clone()
        labels[:, :3] = -100

        out_ours = ours(input_ids, labels=labels)
        out_ref = ref(input_ids, labels=labels)
        assert torch.allclose(out_ours.logits, out_ref.logits, atol=1e-5)
        assert torch.allclose(out_ours.loss, out_ref.loss, atol=1e-5)
        assert torch.allclose(out_ours.aux_loss, out_ref.aux_loss, atol=1e-6)
        if use_moe:
            assert out_ours.aux_loss.item() > 0

        (out_ours.loss + out_ours.aux_loss).backward()
        (out_ref.loss + out_ref.aux_loss).backward()
        assert torch.allclose(ours.lm_head.weight.grad, ref.lm_head.weight.grad, atol=1e-4)


def test_ch10_generation_matches_minimind():
    model_mod = load_module("model")
    gen_mod = load_module("generate")

    # 单元测试各采样过滤器
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    rep = gen_mod.apply_repetition_penalty(logits.clone(), torch.tensor([[3, 0]]), penalty=2.0)
    assert torch.allclose(rep, torch.tensor([[0.5, 2.0, 3.0, 2.0]]))
    tk = gen_mod.top_k_filter(logits.clone(), top_k=2)
    assert torch.isinf(tk[0, 0]) and torch.isinf(tk[0, 1]) and tk[0, 2] == 3.0
    tp = gen_mod.top_p_filter(torch.tensor([[0.0, 0.0, 10.0]]), top_p=0.5)
    assert torch.isinf(tp[0, 0]) and torch.isinf(tp[0, 1]) and tp[0, 2] == 10.0

    # 与 MiniMind 贪心生成逐 token 完全一致，且 use_cache=True 与 False 一致
    cfg = TinyMindConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, vocab_size=100)
    mm_cfg = MiniMindConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                            num_key_value_heads=2, vocab_size=100)
    ours = model_mod.TinyMindForCausalLM(cfg).eval()
    ref = MiniMindForCausalLM(mm_cfg).eval()
    ref.load_state_dict(ours.state_dict())

    prompt = torch.tensor([[1, 15, 23, 42]])
    g_cache = gen_mod.generate(ours, prompt, max_new_tokens=10, do_sample=False, use_cache=True, eos_token_id=99)
    g_nocache = gen_mod.generate(ours, prompt, max_new_tokens=10, do_sample=False, use_cache=False, eos_token_id=99)
    g_mm = ref.generate(prompt, max_new_tokens=10, do_sample=False, use_cache=True, eos_token_id=99)
    assert torch.equal(g_cache, g_nocache)
    assert torch.equal(g_cache, g_mm)
