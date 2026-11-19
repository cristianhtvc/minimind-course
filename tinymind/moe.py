"""
第 9 章：混合专家 MoE（对应 MiniMind: model/model_minimind.py -> MOEFeedForward）

把一个 FFN 换成 N 个「专家」FFN + 一个路由器（gate）：
  1. 路由器给每个 token 打分：scores = softmax(gate(x))，形状 [tokens, num_experts]
  2. 每个 token 只选 top-k 个专家参与计算（稀疏激活）——总参数大，激活参数小
  3. 输出 = Σ_{选中的专家 e} weight_e * expert_e(x)
  4. 辅助负载均衡损失（aux loss）：防止所有 token 都挤到同一个专家上（专家坍塌）
"""
import torch
import torch.nn.functional as F
from torch import nn

from .config import TinyMindConfig
from .ffn import FeedForward, ACT2FN


class MOEFeedForward(nn.Module):
    def __init__(self, config: TinyMindConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([FeedForward(config, intermediate_size=config.moe_intermediate_size)
                                      for _ in range(config.num_experts)])
        self.act_fn = ACT2FN[config.hidden_act]
        self.aux_loss = torch.tensor(0.0)

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)                       # [T, H]，T = batch * seq
        scores = F.softmax(self.gate(x_flat), dim=-1)          # [T, E]

        # TODO(ch09-1): 用 torch.topk 选出每个 token 的 top-k 专家（权重 topk_weight 与下标 topk_idx，sorted=False）
        # >>> SOLUTION
        topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False)
        # <<< SOLUTION

        if self.config.norm_topk_prob:
            if self.config.num_experts_per_tok > 1:
                # k>1：把选中的 k 个权重重新归一化，使其和为 1
                topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
            else:
                # k=1：前向时权重恒为 1.0（归一化后的结果），但用 straight-through 技巧保留对 gate 的梯度
                top1 = torch.topk(F.softmax(self.gate(x_flat.detach()), dim=-1), k=1, dim=-1, sorted=False)[0]
                topk_weight = top1 - top1.detach() + 1.0

        y = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i)                             # [T, k]：哪些 token 的第几个槽位选了专家 i
            if mask.any():
                # TODO(ch09-2): 取出路由到专家 i 的 token，计算 expert(x) * weight，并用 index_add_ 累加回 y
                # >>> SOLUTION
                token_idx = mask.any(dim=-1).nonzero().flatten()
                weight = topk_weight[mask].view(-1, 1)
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
                # <<< SOLUTION
            elif self.training:
                # 没被选中的专家也要「参与」计算图，否则 DDP 会报 unused parameters
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())

        if self.training and self.config.router_aux_loss_coef > 0:
            # TODO(ch09-3): Switch-Transformer 风格负载均衡损失：
            #   load_i = 被分配到专家 i 的 token 比例（one_hot(topk_idx) 在 token 维上求均值）
            #   P_i    = 路由器给专家 i 的平均概率（scores 在 token 维上求均值）
            #   aux = Σ_i load_i * P_i * num_experts * coef
            # >>> SOLUTION
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
            self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
            # <<< SOLUTION
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()
        return y.view(batch_size, seq_len, hidden_dim)
