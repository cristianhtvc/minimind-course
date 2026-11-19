"""
第 18 章：GRPO / CISPO（对应 MiniMind: trainer/train_grpo.py + trainer/rollout_engine.py）

GRPO（DeepSeekMath, 2024）去掉了 PPO 的 critic：
  对同一个 prompt 采样 G 个回答，用「组内相对奖励」作为优势：
      A_i = (r_i - mean(r_1..r_G)) / (std(r_1..r_G) + eps)
  同一条回答的所有 token 共享同一个 A_i。

CISPO（MiniMax-M1, 2025）：不裁剪 token 的梯度，而是裁剪重要性采样权重（并 detach）：
      L = -( sg(clip(ratio, max=ε_high)) * A * logp ) + β * KL
  好处：低概率但关键的 token（如 "However"、"Wait"）不会因为 ratio 越界而被丢弃梯度。
MiniMind 默认 loss_type="cispo"。
"""
import torch
import torch.nn.functional as F

from .ppo import k3_kl


def group_advantages(rewards, num_generations, eps=1e-4):
    """rewards: [B*G]（同一 prompt 的 G 个回答相邻排列）-> advantages: [B*G]"""
    # TODO(ch18-1): view 成 [B, G]，组内求 mean / std(unbiased=False)，repeat_interleave 回 [B*G]，再标准化
    # >>> SOLUTION
    grouped = rewards.view(-1, num_generations)
    mean_r = grouped.mean(dim=1).repeat_interleave(num_generations)
    std_r = grouped.std(dim=1, unbiased=False).repeat_interleave(num_generations)
    return (rewards - mean_r) / (std_r + eps)
    # <<< SOLUTION


def completion_mask_after_eos(completion_ids, pad_mask, eos_token_id):
    """只保留到第一个 eos（含）为止的 token；之后的位置不计入损失。

    completion_ids: [N, R]；pad_mask: [N, R]（1=真实生成的 token）
    """
    is_eos = (completion_ids == eos_token_id) & pad_mask.bool()
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1) - 1, dtype=torch.long, device=completion_ids.device)
    has_eos = is_eos.any(dim=1)
    eos_idx[has_eos] = is_eos.int().argmax(dim=1)[has_eos]
    arange = torch.arange(is_eos.size(1), device=completion_ids.device).expand(is_eos.size(0), -1)
    return ((arange <= eos_idx.unsqueeze(1)) & pad_mask.bool()).int()


def per_token_logps(model, input_ids, n_completion, attention_mask=None):
    """计算序列最后 n_completion 个 token 的 log 概率：[N, n_completion]"""
    logits = model(input_ids, attention_mask=attention_mask, logits_to_keep=n_completion + 1).logits[:, :-1, :]
    target = input_ids[:, -n_completion:]
    return torch.gather(logits.log_softmax(dim=-1), 2, target.unsqueeze(-1)).squeeze(-1)


def grpo_loss(logp, old_logp, ref_logp, advantages, mask, beta=0.1, loss_type="cispo",
              epsilon=0.2, epsilon_high=5.0):
    """logp/old_logp/ref_logp/mask: [N, R]；advantages: [N]。返回标量 loss。"""
    per_token_kl = k3_kl(ref_logp, logp)
    ratio = torch.exp(logp - old_logp)
    adv = advantages.unsqueeze(1)
    # TODO(ch18-2): 两种 per-token 损失
    #   cispo: -( clamp(ratio, max=ε_high).detach() * A * logp - β * KL )
    #   grpo : -( min(ratio * A, clamp(ratio, 1-ε, 1+ε) * A) - β * KL )
    #   然后每条序列在 mask 上求平均，再对所有序列求平均（sequence-level mean）
    # >>> SOLUTION
    if loss_type == "cispo":
        clamped_ratio = torch.clamp(ratio, max=epsilon_high).detach()
        per_token_loss = -(clamped_ratio * adv * logp - beta * per_token_kl)
    else:
        clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
        per_token_loss = -(torch.min(ratio * adv, clipped_ratio * adv) - beta * per_token_kl)
    return ((per_token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)).mean()
    # <<< SOLUTION
