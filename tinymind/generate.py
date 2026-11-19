"""
第 10 章：文本生成（对应 MiniMind: MiniMindForCausalLM.generate）

自回归生成循环：
  for step in range(max_new_tokens):
      logits = model(新 token, past_key_values)[:, -1]   # 只看最后一个位置
      logits = 温度缩放 / 重复惩罚 / top-k / top-p 过滤
      next_token = 采样(softmax(logits)) 或 argmax
      拼接到序列末尾，更新 KV Cache；遇到 eos 停止
"""
import math
import torch
import torch.nn.functional as F


def apply_repetition_penalty(logits, input_ids, penalty: float):
    """对已经出现过的 token 施加惩罚（CTRL 论文的做法）：正 logit 除以 penalty，负 logit 乘以 penalty。

    logits: [B, V]，input_ids: [B, L]。原地修改并返回 logits。
    """
    if penalty == 1.0:
        return logits
    # TODO(ch10-1): 对每个样本，找出 input_ids 中出现过的 token（torch.unique），按规则调整其 logit
    # >>> SOLUTION
    for i in range(input_ids.shape[0]):
        seen = torch.unique(input_ids[i])
        score = logits[i, seen]
        logits[i, seen] = torch.where(score > 0, score / penalty, score * penalty)
    # <<< SOLUTION
    return logits


def top_k_filter(logits, top_k: int):
    """只保留概率最高的 top_k 个 token，其余置为 -inf。logits: [B, V]"""
    if top_k <= 0:
        return logits
    # TODO(ch10-2): 找到每行第 k 大的值作为阈值，小于阈值的位置置为 -inf
    # >>> SOLUTION
    logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float("inf")
    # <<< SOLUTION
    return logits


def top_p_filter(logits, top_p: float):
    """核采样（nucleus sampling）：保留累计概率刚好超过 top_p 的最小 token 集合。logits: [B, V]"""
    if top_p >= 1.0:
        return logits
    # TODO(ch10-3): 降序排序 -> softmax -> cumsum -> 找到累计概率 > top_p 的位置；
    #   注意要把 mask 右移一位（保证至少保留第一个超过阈值的 token），再 scatter 回原顺序
    # >>> SOLUTION
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
    mask = F.pad(mask[..., :-1], (1, 0))                      # 右移一位，第一个永远保留
    logits[mask.scatter(1, sorted_indices, mask)] = -float("inf")
    # <<< SOLUTION
    return logits


@torch.inference_mode()
def generate(model, input_ids, attention_mask=None, max_new_tokens=256, temperature=0.85, top_p=0.85, top_k=50,
             eos_token_id=2, use_cache=True, do_sample=True, repetition_penalty=1.0, num_return_sequences=1,
             streamer=None):
    """与 MiniMind generate 行为一致的生成函数（贪心时两者逐 token 相同）。"""
    if do_sample and not 0 < temperature < math.inf:
        raise ValueError("Sampling temperature must be finite and positive")
    input_ids = input_ids.repeat(num_return_sequences, 1)
    attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
    past_key_values = None
    finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

    for _ in range(max_new_tokens):
        # TODO(ch10-4): 有 KV Cache 时只把「还没进过模型的 token」喂进去：input_ids[:, past_len:]
        # >>> SOLUTION
        past_len = past_key_values[0][0].shape[1] if past_key_values else 0
        outputs = model(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache)
        # <<< SOLUTION
        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1)

        logits = outputs.logits[:, -1].float()
        if do_sample:
            logits = logits / temperature
        logits = apply_repetition_penalty(logits, input_ids, repetition_penalty)
        if do_sample:
            logits = top_k_filter(logits, top_k)
            logits = top_p_filter(logits, top_p)
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

        if eos_token_id is not None:  # 已结束的序列继续填充 eos
            next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        past_key_values = outputs.past_key_values if use_cache else None
        if streamer is not None:
            streamer(next_token)
        if eos_token_id is not None:
            finished |= next_token.squeeze(-1).eq(eos_token_id)
            if finished.all():
                break
    return input_ids
