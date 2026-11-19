"""
Labs —— 你的练习区：把每个 NotImplementedError 换成你的实现。

每个模块对应课程的一章（见仓库根目录 README.md 的章节表）：
    basics.py        第 1-2 章  PyTorch / 语言模型基础
    tokenizer_bpe.py 第 3 章    Byte-Level BPE 分词器
    norm.py          第 4 章    RMSNorm
    attention.py     第 5 章    注意力 / GQA / KV Cache
    rope.py          第 6 章    RoPE / YaRN
    ffn.py           第 7 章    SwiGLU FFN
    model.py         第 8 章    组装完整模型
    moe.py           第 9 章    MoE
    generate.py      第 10 章   采样与生成
    data.py          第 11/13/16 章 数据集
    train_utils.py   第 12 章   训练循环工程
    lora.py          第 14 章   LoRA
    distill.py       第 15 章   知识蒸馏
    dpo.py           第 16 章   DPO
    ppo.py           第 17 章   PPO
    grpo.py          第 18 章   GRPO / CISPO
    agent.py         第 19 章   Tool Calling / Agentic RL
"""
from .config import TinyMindConfig

__all__ = ["TinyMindConfig"]
