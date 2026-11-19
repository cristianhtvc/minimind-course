"""
第 5 章：自注意力（对应 MiniMind: model/model_minimind.py -> repeat_kv / Attention）

本文件包含了现代 LLM 注意力层的全部关键要素：
  1. Q/K/V 线性投影 + 多头拆分
  2. GQA（Grouped Query Attention）：KV 头数少于 Q 头数，用 repeat_kv 广播，省 KV Cache 显存
  3. QK-Norm（Qwen3 引入）：在 RoPE 之前对每个头的 q、k 做 RMSNorm，稳定训练
  4. RoPE 位置编码（第 6 章）
  5. KV Cache：推理时缓存历史 K/V，新 token 只需计算自己的 q/k/v
  6. 因果掩码 + padding 掩码
  7. Flash Attention（F.scaled_dot_product_attention）与手写实现两条路径
"""
import math
import torch
import torch.nn.functional as F
from torch import nn

from .config import TinyMindConfig
from .norm import RMSNorm
from .rope import apply_rotary_pos_emb


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """把 KV 头复制 n_rep 份，使其数量与 Q 头一致。

    x: [bs, slen, n_kv_heads, head_dim] -> [bs, slen, n_kv_heads * n_rep, head_dim]
    第 j 个 KV 头会被第 j*n_rep ... (j+1)*n_rep-1 个 Q 头共享。
    """
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    # TODO(ch05-1): 用 expand + reshape 实现（不要用 for 循环；expand 不拷贝内存）
    raise NotImplementedError("ch05-1: 请完成这里的实现，然后运行对应测试")


class Attention(nn.Module):
    def __init__(self, config: TinyMindConfig):
        super().__init__()
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads   # 每个 KV 头被几个 Q 头共享
        self.head_dim = config.head_dim
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention") and config.flash_attn

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        """
        x: [bsz, seq_len, hidden]
        position_embeddings: (cos, sin)，形状 [seq_len, head_dim]，已经按 start_pos 切好
        past_key_value: (past_k, past_v)，形状 [bsz, past_len, n_kv_heads, head_dim]，或 None
        attention_mask: [bsz, total_len]，1 表示有效 token，0 表示 padding
        返回: output [bsz, seq_len, hidden], present_kv
        """
        bsz, seq_len, _ = x.shape

        # TODO(ch05-2): Q/K/V 投影，并 view 成 [bsz, seq_len, n_heads(或 n_kv_heads), head_dim]，然后做 QK-Norm
        raise NotImplementedError("ch05-2: 请完成这里的实现，然后运行对应测试")

        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        # TODO(ch05-3): KV Cache —— 若有 past_key_value，把历史 K/V 拼接在 seq 维（dim=1）前面；
        #   use_cache=True 时返回拼接后的 (xk, xv) 作为 present，否则返回 None
        raise NotImplementedError("ch05-3: 请完成这里的实现，然后运行对应测试")

        # [bsz, len, heads, head_dim] -> [bsz, heads, len, head_dim]，KV 头先 repeat 到与 Q 头一致
        xq, xk, xv = (xq.transpose(1, 2),
                      repeat_kv(xk, self.n_rep).transpose(1, 2),
                      repeat_kv(xv, self.n_rep).transpose(1, 2))

        use_flash = (self.flash and seq_len > 1 and (not self.is_causal or past_key_value is None)
                     and (attention_mask is None or torch.all(attention_mask == 1)))
        if use_flash:
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0,
                                                    is_causal=self.is_causal)
        else:
            # TODO(ch05-4): 手写缩放点积注意力
            #   1) scores = q @ k^T / sqrt(head_dim)，形状 [bsz, heads, seq_len, kv_len]（用 float32 计算）
            #   2) 因果掩码：只对最后 seq_len 列加上一个上三角(-inf)矩阵（有 KV Cache 时 kv_len > seq_len）
            #   3) padding 掩码：attention_mask 为 0 的位置加上 -1e9
            #   4) softmax -> dropout -> @ v
            raise NotImplementedError("ch05-4: 请完成这里的实现，然后运行对应测试")

        # 合并多头：[bsz, heads, seq, head_dim] -> [bsz, seq, heads*head_dim] -> o_proj
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv
