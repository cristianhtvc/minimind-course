"""
第 4 章：RMSNorm（对应 MiniMind: model/model_minimind.py -> RMSNorm）

RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
与 LayerNorm 相比：不减均值、没有 bias，计算更省，效果相当，是 LLaMA / Qwen 系列的标配。
"""
import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # 可学习的逐维缩放 γ，初始化为 1

    def norm(self, x):
        # TODO(ch04-1): 在最后一维上做 RMS 归一化：x / sqrt(mean(x^2) + eps)
        #   提示：torch.rsqrt(t) = 1 / sqrt(t)；mean 需要 keepdim=True 以便广播
        # >>> SOLUTION
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # <<< SOLUTION

    def forward(self, x):
        # 先转 float32 计算（bf16/fp16 下平方求均值容易溢出/损失精度），再转回原 dtype
        return (self.weight * self.norm(x.float())).type_as(x)
