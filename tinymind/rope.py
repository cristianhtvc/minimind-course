"""
第 6 章：RoPE 旋转位置编码 + YaRN 外推（对应 MiniMind: precompute_freqs_cis / apply_rotary_pos_emb）

核心思想：把 q/k 的每两个维度看作一个复平面上的点，按「位置 m × 频率 θ_i」旋转。
于是 <RoPE(q, m), RoPE(k, n)> 只依赖相对位置 (m - n) —— 用绝对位置的操作得到了相对位置的效果。

MiniMind 使用的是「rotate_half」写法（与 HF LLaMA/Qwen 相同）：
把 head_dim 切成前后两半 [x1, x2]，第 i 维与第 i + d/2 维组成一对进行旋转。
"""
import math
import torch


def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
    """预计算所有位置的 cos / sin 表。

    返回:
        freqs_cos, freqs_sin: 形状均为 [end, dim]
    """
    # TODO(ch06-1): 计算 dim/2 个基础频率 θ_i = 1 / base^(2i/dim), i = 0..dim/2-1
    # >>> SOLUTION
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # <<< SOLUTION
    attn_factor = 1.0

    if rope_scaling is not None:
        # YaRN: f'(i) = f(i) * ((1-γ) + γ/s)，γ ∈ [0,1] 是按维度线性变化的 ramp
        #   - 高频维度（波长远小于训练长度）：γ=0，不缩放（保留局部分辨率）
        #   - 低频维度（波长超过训练长度）：γ=1，频率除以 s（等价于位置插值）
        orig_max = rope_scaling.get("original_max_position_embeddings", 2048)
        factor = rope_scaling.get("factor", 16)
        beta_fast = rope_scaling.get("beta_fast", 32.0)
        beta_slow = rope_scaling.get("beta_slow", 1.0)
        attn_factor = rope_scaling.get("attention_factor", 1.0)
        if end / orig_max > 1.0:
            # 维度 i 在训练长度内转过的圈数 = orig_max / (2π / θ_i)，反解出「转 b 圈」对应的维度
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low = max(math.floor(inv_dim(beta_fast)), 0)
            high = min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            # TODO(ch06-2): 构造 ramp（维度 < low 时为 0，> high 时为 1，中间线性），并按 YaRN 公式缩放 freqs
            # >>> SOLUTION
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)
            # <<< SOLUTION

    # TODO(ch06-3): 计算每个位置 t 与每个频率的乘积（外积），得到角度矩阵 [end, dim/2]，
    #   然后把 cos / sin 在最后一维复制拼接成 [end, dim]（配合 rotate_half 写法），再乘以 attn_factor
    # >>> SOLUTION
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    # <<< SOLUTION
    return freqs_cos, freqs_sin


def rotate_half(x):
    """[x1, x2] -> [-x2, x1]，其中 x1/x2 是最后一维的前/后半部分。"""
    # TODO(ch06-4): 实现 rotate_half
    # >>> SOLUTION
    return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    # <<< SOLUTION


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """对 q, k 施加旋转。

    q: [bsz, seq_len, n_heads, head_dim]
    k: [bsz, seq_len, n_kv_heads, head_dim]
    cos/sin: [seq_len, head_dim]，unsqueeze 后在 heads 维度上广播
    """
    # TODO(ch06-5): q_embed = q * cos + rotate_half(q) * sin（k 同理），注意 unsqueeze 与 dtype
    # >>> SOLUTION
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    # <<< SOLUTION
    return q_embed, k_embed
