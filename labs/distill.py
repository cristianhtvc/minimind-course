"""
第 15 章：知识蒸馏（对应 MiniMind: trainer/train_distillation.py）

白盒蒸馏：学生不仅学习「正确答案」（硬标签 CE），还学习老师在整个词表上的概率分布（软标签 KL）。
  L = α * CE(y, p_s) + (1 - α) * T² * KL(p_t^T || p_s^T)
  - 温度 T > 1 让分布更平滑，暴露老师对「次优答案」的偏好（dark knowledge）
  - 乘 T² 是为了让梯度量级不随 T 改变（Hinton et al., 2015）
"""
import torch
import torch.nn.functional as F


def distillation_loss(student_logits, teacher_logits, temperature=1.0, reduction="batchmean"):
    """KL(teacher || student)，logits 形状 [N, V]。"""
    if student_logits.size(0) == 0 and reduction == "batchmean":
        return student_logits.sum() * 0.0
    # TODO(ch15-1): teacher_probs = softmax(t / T)（不需要梯度），student_log_probs = log_softmax(s / T)，
    #   loss = T^2 * F.kl_div(student_log_probs, teacher_probs, reduction=reduction)
    raise NotImplementedError("ch15-1: 请完成这里的实现，然后运行对应测试")


def distill_step_loss(student_logits, teacher_logits, labels, alpha=0.5, temperature=1.0):
    """完整蒸馏损失（与 MiniMind 一致：只在 label != -100 的位置计算 KL）。

    student_logits/teacher_logits: [B, L, V]（已经去掉最后一位，与 labels[:, 1:] 对齐）
    labels: [B, L]（已经 shift）
    """
    mask = (labels != -100).view(-1)
    ce = F.cross_entropy(student_logits.reshape(-1, student_logits.size(-1)), labels.reshape(-1), ignore_index=-100)
    s = student_logits.reshape(-1, student_logits.size(-1))[mask]
    t = teacher_logits[..., : student_logits.size(-1)].reshape(-1, student_logits.size(-1))[mask]
    kd = distillation_loss(s, t, temperature=temperature)
    return alpha * ce + (1 - alpha) * kd, {"ce": ce.item(), "kd": kd.item()}
