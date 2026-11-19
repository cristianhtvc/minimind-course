"""
第 8 章：组装完整模型（对应 MiniMind: MiniMindBlock / MiniMindModel / MiniMindForCausalLM）

整体数据流：
  input_ids [B, L]
    -> embed_tokens                      [B, L, H]
    -> N × Block:
         h = h + Attn(RMSNorm(h))        (Pre-Norm + 残差)
         h = h + FFN(RMSNorm(h))
    -> final RMSNorm
    -> lm_head                           [B, L, V]   (与 embed_tokens 共享权重)
    -> loss = CE(logits[:, :-1], labels[:, 1:])   (Next Token Prediction)

参数名与 MiniMind 完全一致，因此可以直接 load MiniMind 的 .pth 权重。
"""
from dataclasses import dataclass
from typing import Any, List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .config import TinyMindConfig
from .norm import RMSNorm
from .rope import precompute_freqs_cis
from .attention import Attention
from .ffn import FeedForward
from .moe import MOEFeedForward


@dataclass
class CausalLMOutput:
    loss: Optional[torch.Tensor] = None
    aux_loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    past_key_values: Optional[List[Any]] = None
    hidden_states: Optional[torch.Tensor] = None


class TinyMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: TinyMindConfig):
        super().__init__()
        self.layer_id = layer_id
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        # TODO(ch08-1): Pre-Norm 残差结构
        #   h = h + attn(input_layernorm(h))      （注意 attn 返回 (out, present_kv)）
        #   h = h + mlp(post_attention_layernorm(h))
        raise NotImplementedError("ch08-1: 请完成这里的实现，然后运行对应测试")
        return hidden_states, present_key_value


class TinyMindModel(nn.Module):
    def __init__(self, config: TinyMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([TinyMindBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.head_dim, end=config.max_position_embeddings,
                                                    rope_base=config.rope_theta, rope_scaling=config.rope_scaling)
        # persistent=False：不存进 state_dict（可随时重算），但会跟随 model.to(device)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        batch_size, seq_length = input_ids.shape
        past_key_values = past_key_values or [None] * len(self.layers)
        # 有 KV Cache 时，新 token 的位置从「已缓存长度」开始
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # TODO(ch08-2): 根据 start_pos 从预计算表中切出当前 token 对应的 (cos, sin)
        raise NotImplementedError("ch08-2: 请完成这里的实现，然后运行对应测试")

        presents = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(hidden_states, position_embeddings, past_key_value=past_key_value,
                                           use_cache=use_cache, attention_mask=attention_mask)
            presents.append(present)
        hidden_states = self.norm(hidden_states)
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)],
                       hidden_states.new_zeros(1).squeeze())
        return hidden_states, presents, aux_loss


class TinyMindForCausalLM(nn.Module):
    def __init__(self, config: TinyMindConfig = None):
        super().__init__()
        self.config = config or TinyMindConfig()
        self.model = TinyMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.apply(self._init_weights)
        if self.config.tie_word_embeddings:
            # 权重共享：输入 embedding 与输出投影是同一个矩阵（小模型能省下大量参数）
            self.model.embed_tokens.weight = self.lm_head.weight

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False,
                logits_to_keep=0, labels=None, **kwargs):
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache)
        # logits_to_keep>0 时只算最后若干位置的 logits（推理/RL 时省显存）；0 表示 slice(0, None) 即全部
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            # TODO(ch08-3): Next Token Prediction 损失
            #   位置 t 的 logits 预测位置 t+1 的 token：logits[:, :-1] 对齐 labels[:, 1:]
            #   labels 中为 -100 的位置不计入损失（ignore_index=-100）
            raise NotImplementedError("ch08-3: 请完成这里的实现，然后运行对应测试")
        return CausalLMOutput(loss=loss, aux_loss=aux_loss, logits=logits,
                              past_key_values=past_key_values, hidden_states=hidden_states)

    def generate(self, input_ids, **kwargs):
        from .generate import generate
        return generate(self, input_ids, **kwargs)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """统计参数量（共享权重只算一次，因为 model.parameters() 会去重）。"""
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))
