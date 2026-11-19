"""
第 17 章：PPO 近端策略优化（对应 MiniMind: trainer/train_ppo.py）

RLHF-PPO 需要 4 个模型：
  actor（策略 π_θ，被训练）、critic（价值 V_φ，被训练）、ref（冻结的 SFT 模型，算 KL）、reward model（打分）

单步流程：
  1. rollout：actor 对 prompt 采样回答，记录 old_logp、old_values
  2. 奖励：reward model 给整条回答打分，只加在最后一个 token 上（稀疏奖励）
  3. GAE：从后往前算每个 token 的优势 A_t 与回报 R_t = A_t + V_t
  4. 多轮小批量更新：clip 的策略损失 + clip 的价值损失 + KL(ref) 惩罚
"""
import torch


def compute_gae(token_rewards, values, gamma=1.0, lam=0.95):
    """广义优势估计 GAE（Schulman et al., 2016）。

    token_rewards, values: [B, T]
    δ_t = r_t + γ V_{t+1} - V_t            （最后一个位置 V_{T} = 0）
    A_t = δ_t + γλ A_{t+1}                 （从后往前递推）
    返回 advantages [B, T], returns = advantages + values
    """
    B, T = values.shape
    lastgaelam = torch.zeros(B, device=values.device, dtype=values.dtype)
    advs_rev = []
    # TODO(ch17-1): 按上式从 t = T-1 倒序递推到 0
    raise NotImplementedError("ch17-1: 请完成这里的实现，然后运行对应测试")
    advantages = torch.stack(advs_rev[::-1], dim=1)
    returns = advantages + values
    return advantages, returns


def masked_whiten(x, mask, eps=1e-8):
    """在 mask 有效位置上做标准化（零均值单位方差），无效位置置 0。"""
    mean = (x * mask).sum() / mask.sum().clamp(min=1)
    var = ((x - mean) ** 2 * mask).sum() / mask.sum().clamp(min=1)
    return (x - mean) * torch.rsqrt(var + eps) * mask


def masked_mean(x, mask):
    return (x * mask).sum() / mask.sum().clamp(min=1)


def ppo_policy_loss(logp, old_logp, advantages, mask, clip_eps=0.2):
    """PPO-Clip 策略损失。

    ratio = exp(logp - old_logp)
    L = mean( max(-A * ratio, -A * clip(ratio, 1-ε, 1+ε)) )   （等价于 -min(...)）
    返回 loss, clipfrac（被裁剪的 token 比例，用于监控）
    """
    # TODO(ch17-2): 实现 PPO-Clip（注意只在 mask 位置上求平均）
    raise NotImplementedError("ch17-2: 请完成这里的实现，然后运行对应测试")
    return loss, clipfrac


def ppo_value_loss(values, old_values, returns, mask, clip_value=0.2):
    """价值函数损失（同样做 clip，防止 critic 一步走太远）：
    L_V = 0.5 * mean( max((V - R)^2, (clip(V, V_old-c, V_old+c) - R)^2) )
    """
    # TODO(ch17-3)
    raise NotImplementedError("ch17-3: 请完成这里的实现，然后运行对应测试")


def k3_kl(ref_logp, logp):
    """KL(π || π_ref) 的 k3 无偏低方差估计（John Schulman 博客）：exp(r) - r - 1，r = ref_logp - logp。恒 ≥ 0。"""
    r = ref_logp - logp
    return torch.exp(r) - r - 1.0
