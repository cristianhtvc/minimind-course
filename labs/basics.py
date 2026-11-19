"""
第 1-2 章：PyTorch 与语言模型基础（对应 MiniMind 中无处不在的 softmax / cross_entropy）

在动手搭 Transformer 之前，先用最朴素的「二元语法（Bigram）语言模型」把
「预测下一个 token -> 交叉熵 -> 反向传播 -> 采样」这条主线跑通。
MiniMind 的所有训练阶段，本质上都是在这个主线上叠加更复杂的模型和数据。
"""
import math
import torch
import torch.nn as nn


def stable_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """数值稳定的 softmax：先减去最大值再 exp，防止 exp 溢出。"""
    # TODO(ch01-1)
    raise NotImplementedError("ch01-1: 请完成这里的实现，然后运行对应测试")


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """手写交叉熵：-mean(log softmax(logits)[target])，跳过 ignore_index。

    logits: [N, V]，targets: [N]
    """
    # TODO(ch02-1): 用 log-sum-exp 技巧计算 log_softmax，再取出 target 位置，忽略 ignore_index 后求平均
    raise NotImplementedError("ch02-1: 请完成这里的实现，然后运行对应测试")


def perplexity(mean_ce_loss: float) -> float:
    """困惑度 PPL = exp(平均交叉熵)。直觉：模型在每一步平均「犹豫」于多少个候选 token 之间。"""
    # TODO(ch02-2)
    raise NotImplementedError("ch02-2: 请完成这里的实现，然后运行对应测试")


class BigramLM(nn.Module):
    """最简单的语言模型：下一个 token 的分布只取决于当前 token。参数就是一张 [V, V] 的查找表。"""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.table(idx)  # [B, L, V]
        loss = None
        if targets is not None:
            loss = cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -1:])
            probs = stable_softmax(logits[:, -1])
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx
