"""
第 3 章：从零实现 Byte-Level BPE 分词器（对应 MiniMind: trainer/train_tokenizer.py）

MiniMind 用 HuggingFace `tokenizers` 库训练了一个 6400 词表的 Byte-Level BPE。
这里我们用 ~150 行纯 Python 把它的核心算法手写一遍（思路参考 karpathy/minbpe）：

  训练：
    1. 文本 -> 预切分（按空格/标点/数字切成 chunk，chunk 之间不合并）
    2. 每个 chunk -> UTF-8 字节序列（初始词表 = 256 个字节，永远不会 OOV）
    3. 反复统计「相邻 token 对」的出现频率，把最高频的一对合并成新 token，直到词表达到 vocab_size
  编码：按训练时学到的合并顺序（rank）依次合并
  解码：token -> 字节拼接 -> UTF-8 解码
"""
import re
from typing import Dict, List, Tuple

# 预切分正则（GPT-2 风格的简化版；Python 内置 re 不支持 \p{L}，因此用字符类近似）
SPLIT_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?\d| ?[^\sA-Za-z\d]+|\s+(?!\S)|\s+"""


def get_stats(ids: List[int], counts: Dict[Tuple[int, int], int] = None) -> Dict[Tuple[int, int], int]:
    """统计相邻 pair 的出现次数。例：[1,2,3,1,2] -> {(1,2):2, (2,3):1, (3,1):1}"""
    counts = {} if counts is None else counts
    # TODO(ch03-1): 遍历相邻元素对并计数
    # >>> SOLUTION
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    # <<< SOLUTION
    return counts


def merge(ids: List[int], pair: Tuple[int, int], idx: int) -> List[int]:
    """把 ids 中所有出现的 pair 替换为新 token idx。例：merge([1,2,3,1,2], (1,2), 4) -> [4,3,4]"""
    # TODO(ch03-2): 从左到右扫描，遇到 pair 就输出 idx 并跳过 2 个位置
    # >>> SOLUTION
    new_ids, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids
    # <<< SOLUTION


class ByteLevelBPE:
    def __init__(self, special_tokens: List[str] = None):
        self.merges: Dict[Tuple[int, int], int] = {}          # (a, b) -> 新 token id，插入顺序即合并优先级
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.special_tokens: Dict[str, int] = {}
        self._pending_special = special_tokens or []
        self.pattern = re.compile(SPLIT_PATTERN)

    # ------------------------------------------------------------------ 训练
    def train(self, text: str, vocab_size: int, verbose: bool = False):
        assert vocab_size >= 256 + len(self._pending_special)
        num_merges = vocab_size - 256 - len(self._pending_special)
        chunks = self.pattern.findall(text)
        ids_list = [list(ch.encode("utf-8")) for ch in chunks]

        for i in range(num_merges):
            # TODO(ch03-3): BPE 训练主循环
            #   1) 对所有 chunk 统计 pair 频次（chunk 之间不跨界）
            #   2) 取出现次数最多的 pair（没有 pair 可合并时提前结束）
            #   3) 分配新 id = 256 + i，在所有 chunk 中执行 merge
            #   4) 记录 self.merges[pair] = new_id，self.vocab[new_id] = vocab[a] + vocab[b]
            # >>> SOLUTION
            stats = {}
            for ids in ids_list:
                get_stats(ids, stats)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            new_id = 256 + i
            ids_list = [merge(ids, pair, new_id) for ids in ids_list]
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            # <<< SOLUTION
            if verbose and i % 50 == 0:
                print(f"merge {i + 1}/{num_merges}: {pair} -> {new_id} ({self.vocab[new_id]!r}) x{stats[pair]}")

        # 特殊 token 放在词表最后（MiniMind 是放在最前：<|endoftext|>=0, <|im_start|>=1, <|im_end|>=2）
        next_id = max(self.vocab) + 1
        for tok in self._pending_special:
            self.special_tokens[tok] = next_id
            self.vocab[next_id] = tok.encode("utf-8")
            next_id += 1
        return self

    # ------------------------------------------------------------------ 编码
    def _encode_chunk(self, chunk_bytes: bytes) -> List[int]:
        ids = list(chunk_bytes)
        while len(ids) >= 2:
            # TODO(ch03-4): 在当前 ids 的所有相邻 pair 中，找「最早被学到」（merges 中 id 最小）的那个进行合并；
            #   如果没有任何 pair 在 merges 中，结束
            # >>> SOLUTION
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
            # <<< SOLUTION
        return ids

    def encode(self, text: str) -> List[int]:
        if not self.special_tokens:
            parts = [text]
        else:  # 先按特殊 token 切开，特殊 token 直接映射，不参与 BPE
            special_pattern = "(" + "|".join(re.escape(s) for s in self.special_tokens) + ")"
            parts = re.split(special_pattern, text)
        ids = []
        for part in parts:
            if part in self.special_tokens:
                ids.append(self.special_tokens[part])
            elif part:
                for chunk in self.pattern.findall(part):
                    ids.extend(self._encode_chunk(chunk.encode("utf-8")))
        return ids

    # ------------------------------------------------------------------ 解码
    def decode(self, ids: List[int]) -> str:
        # errors="replace"：单个 token 可能只是一个汉字 UTF-8 编码的一部分，流式输出时要特别处理（见第 3 章）
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def __len__(self):
        return len(self.vocab)
