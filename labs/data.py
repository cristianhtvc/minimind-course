"""
第 11 / 13 / 16 章：数据集构造（对应 MiniMind: dataset/lm_dataset.py）

三种数据格式（与 MiniMind 一致）：
  预训练  {"text": "..."}
  SFT     {"conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
  DPO     {"chosen": [...messages...], "rejected": [...messages...]}

核心知识点：
  - 预训练：整段文本都算 loss，只有 padding 位置被忽略（label = -100）
  - SFT：只有 assistant 回复部分算 loss（用户问题、system、模板标记都不算）
  - DPO：同样只对 assistant 部分求 log-prob，用 0/1 的 loss_mask 表示
"""
import json
import random
from typing import List

import torch
from torch.utils.data import Dataset

IGNORE_INDEX = -100


def load_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ============================================================ 预训练
class PretrainDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        """data: jsonl 路径，或已加载的 [{"text": ...}] 列表"""
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_jsonl(data) if isinstance(data, str) else data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        # TODO(ch11-1): 构造预训练样本
        #   1) tokenize 文本（不加特殊 token，截断到 max_length-2）
        #   2) 前后加上 bos / eos
        #   3) 右侧用 pad 补齐到 max_length
        #   4) labels = input_ids 的拷贝，但 pad 位置设为 -100
        raise NotImplementedError("ch11-1: 请完成这里的实现，然后运行对应测试")
        return input_ids, labels


# ============================================================ SFT
SYSTEM_PROMPTS = [
    "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
    "你是minimind，一个小巧但有用的语言模型。",
    "You are a helpful AI assistant.",
    "You are minimind, a small but useful language model.",
]


def pre_processing_chat(conversations, add_system_ratio=0.2):
    """以一定概率随机插入 system prompt，让模型对「有/无 system」都鲁棒（数据增强）。"""
    if any(conv.get("tools") for conv in conversations):
        return conversations
    if conversations[0].get("role") != "system" and random.random() < add_system_ratio:
        return [{"role": "system", "content": random.choice(SYSTEM_PROMPTS)}] + list(conversations)
    return conversations


def post_processing_chat(prompt_content, empty_think_ratio=0.2, remove_empty_think=None):
    """chat_template 默认会给每个 assistant 回复插入空的 <think>\\n\\n</think>\\n\\n；
    以 80% 概率删除它，让模型同时见过「有空思考块」和「没有思考块」两种格式（自适应思考的基础）。"""
    if "<think>\n\n</think>\n\n" in prompt_content:
        if remove_empty_think is None:
            remove_empty_think = random.random() > empty_think_ratio
        if remove_empty_think:
            prompt_content = prompt_content.replace("<think>\n\n</think>\n\n", "")
    return prompt_content


def generate_sft_labels(input_ids: List[int], bos_id: List[int], eos_id: List[int], max_length: int,
                        ignore_value=IGNORE_INDEX, keep_value=None) -> List[int]:
    """只保留 assistant 回复部分的 label。

    bos_id: "<|im_start|>assistant\\n" 的 token 序列（回复开始标记）
    eos_id: "<|im_end|>\\n" 的 token 序列（回复结束标记，结束标记本身也要学！否则模型不会停）
    keep_value=None 时保留原 token id（SFT labels）；keep_value=1 且 ignore_value=0 时得到 0/1 loss mask（DPO）。
    """
    labels = [ignore_value] * len(input_ids)
    i = 0
    while i < len(input_ids):
        # TODO(ch13-1): 扫描 input_ids：
        #   找到 bos_id 子序列 -> 从其后一位 start 开始，向后找 eos_id 子序列的位置 end；
        #   把 [start, end + len(eos_id)) 范围（且 < max_length）内的 label 设为 input_ids[j]（或 keep_value）；
        #   然后 i 跳到 end + len(eos_id)；没匹配到 bos 则 i += 1
        raise NotImplementedError("ch13-1: 请完成这里的实现，然后运行对应测试")
    return labels


class SFTDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=1024, augment=True):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
        self.samples = load_jsonl(data) if isinstance(data, str) else data
        self.bos_id = tokenizer(f"{tokenizer.bos_token}assistant\n", add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f"{tokenizer.eos_token}\n", add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        messages, tools = [], None
        for message in conversations:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                message["tool_calls"] = json.loads(message["tool_calls"])
            messages.append(message)
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, tools=tools)

    def __getitem__(self, index):
        conversations = self.samples[index]["conversations"]
        if self.augment:
            conversations = pre_processing_chat(conversations)
        prompt = self.create_chat_prompt(conversations)
        prompt = post_processing_chat(prompt, remove_empty_think=None if self.augment else True)
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = generate_sft_labels(input_ids, self.bos_id, self.eos_id, self.max_length)
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


# ============================================================ DPO
class DPODataset(Dataset):
    def __init__(self, data, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_jsonl(data) if isinstance(data, str) else data
        self.bos_id = tokenizer(f"{tokenizer.bos_token}assistant\n", add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f"{tokenizer.eos_token}\n", add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def _encode(self, messages):
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prompt = post_processing_chat(prompt, remove_empty_think=True)
        ids = self.tokenizer(prompt, truncation=True, max_length=self.max_length, padding="max_length").input_ids
        mask = generate_sft_labels(ids, self.bos_id, self.eos_id, self.max_length, ignore_value=0, keep_value=1)
        # 输入 x 与目标 y 错开一位；mask 与 y 对齐
        return (torch.tensor(ids[:-1]), torch.tensor(ids[1:]), torch.tensor(mask[1:]))

    def __getitem__(self, index):
        sample = self.samples[index]
        x_c, y_c, m_c = self._encode(sample["chosen"])
        x_r, y_r, m_r = self._encode(sample["rejected"])
        return {"x_chosen": x_c, "y_chosen": y_c, "mask_chosen": m_c,
                "x_rejected": x_r, "y_rejected": y_r, "mask_rejected": m_r}
