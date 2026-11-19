"""
第 7 章：前馈网络 SwiGLU（对应 MiniMind: model/model_minimind.py -> FeedForward）

FFN(x) = down( act(gate(x)) ⊙ up(x) )
  - gate/up: hidden -> intermediate（升维）
  - down:    intermediate -> hidden（降维）
  - act = SiLU(x) = x * sigmoid(x)；SiLU 门控 + 逐元素乘 = SwiGLU
"""
import torch.nn.functional as F
from torch import nn

from .config import TinyMindConfig

ACT2FN = {"silu": F.silu, "gelu": F.gelu, "relu": F.relu}


class FeedForward(nn.Module):
    def __init__(self, config: TinyMindConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        # TODO(ch07-1): 实现 SwiGLU：down( act(gate(x)) * up(x) )
        # >>> SOLUTION
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        # <<< SOLUTION
