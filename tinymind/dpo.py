"""
第 16 章：DPO 直接偏好优化（对应 MiniMind: trainer/train_dpo.py）

L_DPO = -log σ( β * [ (log π(y_w|x) - log π_ref(y_w|x)) - (log π(y_l|x) - log π_ref(y_l|x)) ] )
  - y_w: chosen（更好的回答），y_l: rejected（更差的回答）
  - π_ref: 冻结的参考模型（通常是 SFT 模型），起到 KL 约束的作用
  - 只需要 actor + ref 两个模型，不需要 reward model 和 critic
"""
import torch
import torch.nn.functional as F


def logits_to_log_probs(logits, labels):
    """取出每个位置上「目标 token」的 log 概率。

    logits: [B, L, V]，labels: [B, L] -> [B, L]
    """
    # TODO(ch16-1): log_softmax 后用 torch.gather 在词表维取出 labels 对应的值
    # >>> SOLUTION
    log_probs = F.log_softmax(logits, dim=2)
    return torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    # <<< SOLUTION


def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    """ref/policy_log_probs: [2B, L]，前 B 行是 chosen，后 B 行是 rejected；mask: [2B, L]"""
    # TODO(ch16-2):
    #   1) 用 mask 把每条序列的 token log-prob 求和，得到序列级 log-prob [2B]
    #   2) 拆成 chosen / rejected 两半
    #   3) logits = (π_c - π_r) - (ref_c - ref_r)
    #   4) loss = -logsigmoid(β * logits).mean()
    # >>> SOLUTION
    ref_log_probs = (ref_log_probs * mask).sum(dim=1)
    policy_log_probs = (policy_log_probs * mask).sum(dim=1)
    batch_size = ref_log_probs.shape[0]
    chosen_ref, reject_ref = ref_log_probs[:batch_size // 2], ref_log_probs[batch_size // 2:]
    chosen_pi, reject_pi = policy_log_probs[:batch_size // 2], policy_log_probs[batch_size // 2:]
    logits = (chosen_pi - reject_pi) - (chosen_ref - reject_ref)
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()
    # <<< SOLUTION


def dpo_batch_loss(model, ref_model, batch, beta=0.1, device="cpu"):
    """给 DPODataset 的一个 batch 计算 DPO 损失。"""
    x = torch.cat([batch["x_chosen"], batch["x_rejected"]], dim=0).to(device)
    y = torch.cat([batch["y_chosen"], batch["y_rejected"]], dim=0).to(device)
    mask = torch.cat([batch["mask_chosen"], batch["mask_rejected"]], dim=0).to(device)
    with torch.no_grad():
        ref_log_probs = logits_to_log_probs(ref_model(x).logits, y)
    out = model(x)
    policy_log_probs = logits_to_log_probs(out.logits, y)
    loss = dpo_loss(ref_log_probs, policy_log_probs, mask, beta) + out.aux_loss
    return loss, {}
