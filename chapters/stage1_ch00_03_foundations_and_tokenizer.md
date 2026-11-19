# 第一阶段（第 0–3 章）：全景导览、语言模型数学基础与 Byte-Level BPE 分词器

> **本阶段目标**：先在脑中建立 MiniMind 全项目的「代码地图」，掌握自回归语言模型最底层的数学主线（Softmax $\to$ Cross-Entropy $\to$ Perplexity），然后亲手从零实现一个 Byte-Level BPE 分词器，并彻底读懂 MiniMind 的 `tokenizer_config.json` 与 Jinja2 聊天模板。
>
> **配套代码与练习**：
> - 参考实现：[tinymind/basics.py](../tinymind/basics.py)、[tinymind/tokenizer_bpe.py](../tinymind/tokenizer_bpe.py)
> - 动手练习：[labs/basics.py](../labs/basics.py)（`TODO(ch01-1)` ~ `TODO(ch02-2)`）、[labs/tokenizer_bpe.py](../labs/tokenizer_bpe.py)（`TODO(ch03-1)` ~ `TODO(ch03-4)`）
> - 对照 MiniMind 源码：[minimind/trainer/train_tokenizer.py](../minimind/trainer/train_tokenizer.py)、[minimind/model/tokenizer_config.json](../minimind/model/tokenizer_config.json)
> - 闯关评测：`python tools/grade.py ch01`、`python tools/grade.py ch03`

---

## 第 0 章：MiniMind 全景地图 —— 22 个文件分别在干什么？

很多人第一次打开 `minimind` 仓库时会觉得无从下手：既有 `model/`，又有 `dataset/`，还有 `trainer/` 下 10 个不同的训练脚本和 `scripts/` 下的服务脚本。其实，**整个 MiniMind 只有一条主线**：

```
原始文本 ──(1.分词器)──▶ Token IDs ──(2.模型架构)──▶ Logits ──(3.损失函数/优化器)──▶ 演进的权重
```

把所有 22 个 Python 文件按职责归类，其实只有 **5 个核心模块**：

| 模块分类 | MiniMind 源文件 | 行数 | 一句话说明它在整个系统中的角色 | 对应课程章节 |
| :--- | :--- | :---: | :--- | :---: |
| **① 分词与数据** | [trainer/train_tokenizer.py](../minimind/trainer/train_tokenizer.py) | 189 | 训练 6400 词表的 Byte-Level BPE 分词器，写入特殊 Token 和 Jinja2 `chat_template` | 第 3 章 |
| | [dataset/lm_dataset.py](../minimind/dataset/lm_dataset.py) | 260 | 定义 5 种 Dataset：`PretrainDataset`、`SFTDataset`、`DPODataset`、`RLAIFDataset`、`AgentRLDataset` | 第 11, 13, 16, 18, 19 章 |
| **② 模型架构** | [model/model_minimind.py](../minimind/model/model_minimind.py) | 292 | **全项目的灵魂**：`MiniMindConfig`、`RMSNorm`、RoPE/YaRN、`Attention`(GQA+KV Cache)、`FeedForward`(SwiGLU)、`MOEFeedForward`、`MiniMindForCausalLM`、`generate` | 第 4–10 章 |
| | [model/model_lora.py](../minimind/model/model_lora.py) | 70 | 纯手写 `LoRA` 模块、动态挂载 `apply_lora`、存取与合并 `merge_lora` | 第 14 章 |
| **③ 训练基础设施** | [trainer/trainer_utils.py](../minimind/trainer/trainer_utils.py) | 208 | 余弦学习率 `get_lr`、DDP 初始化、原子化断点续训 `lm_checkpoint`、`SkipBatchSampler`、`LMForRewardModel`、`safe_math_eval` | 第 12, 17, 19 章 |
| | [trainer/rollout_engine.py](../minimind/trainer/rollout_engine.py) | 224 | RL 在线采样引擎抽象：支持原生 PyTorch (`TorchRolloutEngine`) 与 SGLang (`SGLangRolloutEngine`) | 第 18 章 |
| **④ 七大训练阶段** | [trainer/train_pretrain.py](../minimind/trainer/train_pretrain.py) | 163 | **阶段 1（必选）**：在海量无标注文本上做 Next-Token Prediction，学会语言规律与世界知识 | 第 11–12 章 |
| | [trainer/train_full_sft.py](../minimind/trainer/train_full_sft.py) | 164 | **阶段 2（必选）**：全参数指令微调，只对 Assistant 回复部分计算交叉熵，学会对话、思考与工具格式 | 第 13 章 |
| | [trainer/train_lora.py](../minimind/trainer/train_lora.py) | 178 | **阶段 3（可选）**：冻结主干，只训练低秩矩阵 $A, B$，低成本注入医疗/自我认知等垂域能力 | 第 14 章 |
| | [trainer/train_distillation.py](../minimind/trainer/train_distillation.py) | 240 | **阶段 4（可选）**：白盒知识蒸馏，用大模型 logits 的软分布（KL 散度）指导小模型 | 第 15 章 |
| | [trainer/train_dpo.py](../minimind/trainer/train_dpo.py) | 218 | **阶段 5（可选）**：离线偏好对齐（DPO），无需奖励模型，直接拉大 `chosen` 与 `rejected` 的概率比 | 第 16 章 |
| | [trainer/train_ppo.py](../minimind/trainer/train_ppo.py) | 447 | **阶段 6A（可选）**：经典在线 RL（Actor + Critic + Ref + Reward 四模型 + GAE 广义优势估计） | 第 17 章 |
| | [trainer/train_grpo.py](../minimind/trainer/train_grpo.py) | 326 | **阶段 6B（可选）**：DeepSeek GRPO / MiniMax CISPO，去掉 Critic，用组内相对奖励做优势 | 第 18 章 |
| | [trainer/train_agent.py](../minimind/trainer/train_agent.py) | 488 | **阶段 6C（进阶）**：多轮 Tool-Use Agent 强化学习，模型在沙箱中多轮调用工具并根据结果拿奖励 | 第 19 章 |
| **⑤ 推理、评测与部署** | [eval_llm.py](../minimind/eval_llm.py) | 98 | 命令行对话测试，支持加载 `.pth`、挂载 LoRA、切换 `open_thinking` 与流式输出 | 第 10, 20 章 |
| | [scripts/eval_toolcall.py](../minimind/scripts/eval_toolcall.py) | 240 | 多轮工具调用闭环测试脚本 | 第 19 章 |
| | [scripts/convert_model.py](../minimind/scripts/convert_model.py) | 144 | 将 MiniMind `.pth` 转换为 HuggingFace `LlamaForCausalLM` / `Qwen3` 格式，以便接入 vLLM/Ollama | 第 20 章 |
| | [scripts/serve_openai_api.py](../minimind/scripts/serve_openai_api.py) | 276 | FastAPI 实现的 OpenAI `/v1/chat/completions` 兼容接口（支持流式、`reasoning_content`、`tool_calls`） | 第 20 章 |
| | [scripts/web_demo.py](../minimind/scripts/web_demo.py) | 424 | Streamlit 网页聊天界面 | 第 20 章 |

> **💡 为什么推荐按本课程的 6 个阶段顺序学习？**
> 因为 MiniMind 的 7 个 `train_*.py` 脚本中，**80% 的工程骨架代码是完全相同的**（DDP 初始化、混合精度、梯度累积、断点续训都在第 12 章一次讲透），真正变化的是 **数据掩码怎么打** 和 **Loss 怎么算**。先吃透 292 行的 `model_minimind.py` 和 163 行的 `train_pretrain.py`，后面的 SFT、LoRA、蒸馏、DPO、PPO、GRPO 每次只需要学几十行核心增量代码！

---

## 第 1–2 章：语言模型数学基础 —— Softmax、交叉熵与困惑度

### 1.1 自回归语言模型到底在算什么？

不管模型有 64M 还是 671B 参数，自回归（Autoregressive）语言模型做的事情只有一件：给定前面的 token 序列 $x_1, x_2, \dots, x_t$，预测下一个 token $x_{t+1}$ 在词表 $\mathcal{V}$（比如大小 $V=6400$）上的概率分布：

$$P(x_1, x_2, \dots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_1, \dots, x_{t-1})$$

神经网络的最后一层（`lm_head`）对每个位置输出一个未归一化的实数向量 $\mathbf{z} \in \mathbb{R}^V$，称为 **logits**。

### 1.2 数值稳定的 Softmax 与 LogSumExp

把 logits $\mathbf{z}$ 变成概率分布 $\mathbf{p}$ 用的是 Softmax：

$$p_i = \frac{e^{z_i}}{\sum_{j=1}^V e^{z_j}}$$

**工程陷阱**：如果某个 $z_i = 100$（在 fp16 下甚至只要 $z_i > 11$），$e^{z_i}$ 就会直接上溢成 `inf`，`inf / inf` 变成 `NaN`！
**解决办法**：利用恒等式，分子分母同乘 $e^{-M}$（其中 $M = \max_j z_j$）：

$$\text{softmax}(\mathbf{z})_i = \frac{e^{z_i - M}}{\sum_{j=1}^V e^{z_j - M}}$$

因为 $z_i - M \le 0$，所以 $e^{z_i - M} \in (0, 1]$，**永远不会上溢**，且分母中至少有一项等于 $e^0 = 1$，**分母永远不会下溢为 0**。

同理，在算交叉熵损失时，我们需要 $\log p_i$。**千万不要先算 `softmax` 再取 `log`**（当 $p_i$ 极小时会下溢为 0，$\log 0 = -\infty$），而是直接用 **LogSumExp** 化简：

$$\log p_i = z_i - \log \sum_{j=1}^V e^{z_j} = z_i - \left(M + \log \sum_{j=1}^V e^{z_j - M}\right)$$

### 1.3 交叉熵（Cross-Entropy）与 `ignore_index = -100`

如果位置 $t$ 的真实下一个 token 是 $y_t \in \{0, \dots, V-1\}$，那么模型在这个位置上的**负对数似然（即交叉熵损失）**就是：

$$\ell_t = -\log p_{y_t}$$

- 如果模型100%猜对（$p_{y_t} = 1$），损失 $\ell_t = -\log 1 = 0$。
- 如果模型在 $V=6400$ 个词里完全瞎猜（均匀分布 $p_{y_t} = 1/6400$），损失 $\ell_t = \ln(6400) \approx 8.764$。
  - **记住这个数字 `8.76`！** 当你启动 MiniMind 预训练时，第 1 步打印出来的 loss 一定在 $8.7 \sim 8.9$ 左右——如果远大于它，说明初始化方差太大；如果等于 `NaN`，说明学习率或精度溢出。

**为什么 MiniMind 到处都在用 `-100`？**
在一个 batch 里，不同句子长度不同，短句后面会补 `pad_token_id`；在 SFT 时，用户的提问（`user` 部分）也不希望模型去背诵。PyTorch 的 `F.cross_entropy(..., ignore_index=-100)` 规定：**凡是 `label == -100` 的位置，既不产生梯度，也不计入求平均的分母**。
对应 MiniMind 源码 [model/model_minimind.py#L255-L257](../minimind/model/model_minimind.py#L255-L257)：
```python
if labels is not None:
    x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
    loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
```

### 1.4 困惑度（Perplexity, PPL）与 BPB（Bits Per Byte）

- **困惑度 $\text{PPL} = e^{\mathcal{L}_{\text{CE}}}$**：如果平均交叉熵损失是 $2.0$，那么 $\text{PPL} = e^2 \approx 7.39$，直观含义是「模型在每一步平均就像在 7.39 个等可能的候选词里做选择」。
- **拓展知识：为什么跨模型不能直接比 PPL？**（MiniMind README 特别强调了这一点）
  因为 PPL 是**按 token 平均**的！如果模型 A 的词表很大（15 万），一个汉字是 1 个 token；模型 B 的词表很小（或者按字节分词），一个汉字被拆成 2~3 个 token。拆分越细，单个 token 的预测往往越容易，PPL 看起来更低，但这不代表模型更强。
  跨分词器比较时，应该用 **BPB（Bits Per Byte）**：
  $$\text{BPB} = \frac{\text{总交叉熵（以 }\log_2\text{ 计）}}{\text{文本的 UTF-8 字节总数}} = \frac{\mathcal{L}_{\text{CE}} \times N_{\text{tokens}}}{\ln 2 \times N_{\text{bytes}}}$$

---

## 第 3 章：Byte-Level BPE 分词器与 ChatML 模板

### 3.1 为什么 MiniMind 选择 6400 词表？（小模型的参数账本）

主流大模型（如 Qwen2.5/Qwen3 的 151,643、Llama-3 的 128,000）动辄十几个万的词表。为什么 MiniMind 只用 **6,400**？
让我们算一笔账：词嵌入层 `embed_tokens` 的权重形状是 `[vocab_size, hidden_size]`。
- 当 `hidden_size = 768` 时：
  - 如果像 Qwen 一样用 `151,643` 词表：仅 Embedding 层就有 $151643 \times 768 \approx \mathbf{1.16\text{ 亿（116M）}}$ 参数！比 MiniMind 整个主干网络（59M）还要大两倍！
  - 用 `6,400` 词表：Embedding 层只有 $6400 \times 768 \approx \mathbf{4.91\text{M}}$ 参数（且与 `lm_head` 共享权重），把 92% 的参数预算都留给了真正负责思考和推理的 Transformer 层。

### 3.2 Byte-Level BPE 算法原理（对应 `tinymind/tokenizer_bpe.py`）

BPE（Byte Pair Encoding）的核心思想只有一句话：**从 256 个基础 UTF-8 字节出发，每次把语料中出现频率最高的一对相邻 token 合并成一个新 token，重复直到词表达到 `vocab_size`。**

1. **为什么以 Byte（字节）为基础？**
   任何人类文字、Emoji、代码在 UTF-8 下都是 `0..255` 的字节序列。以 256 个字节作为初始词表（`0..255`），意味着**哪怕遇到从未见过的生僻字或乱码，最差情况也是退化成单字节输出，永远不会出现 `<unk>`（Out-Of-Vocabulary）！**
2. **训练过程（4 步循环）**：
   - **Step 1（预切分 Pre-tokenization）**：先用正则表达式把文本按单词、数字、标点、空白切成小段（chunks），防止跨词合并（比如不会把 `"dog."` 合并成一个 token）。
   - **Step 2（统计相邻频次 `get_stats`）**：统计所有 chunk 内相邻 pair `(a, b)` 的出现次数。
   - **Step 3（选最高频合并 `merge`）**：找出频率最高的 `pair = (a, b)`，分配新 ID `new_id = 256 + step`，把所有 chunk 中出现的 `(a, b)` 替换为 `new_id`。
   - **Step 4（更新词表）**：`vocab[new_id] = vocab[a] + vocab[b]`（字节拼接）。
3. **编码（Encode）过程**：
   不能贪心合并当前最长的，而必须**严格按照训练时学到的合并优先级（即 `merges[pair]` 越小越优先）依次合并**。

### 3.3 对照 MiniMind 源码：`trainer/train_tokenizer.py` 逐段拆解

打开 [minimind/trainer/train_tokenizer.py](../minimind/trainer/train_tokenizer.py)，重点看以下 4 个设计细节：

1. **预留 36 个 Special / Buffer Tokens（第 49–63 行）**：
   - `special_tokens_list`（21 个）：包含 `<|endoftext|>` (ID=0, 用作 pad/unk)、`<|im_start|>` (ID=1, 用作 bos)、`<|im_end|>` (ID=2, 用作 eos)，以及为多模态（MiniMind-V 视觉/语音）预留的 `<|vision_start|>`、`<|image_pad|>`、`<|audio_pad|>` 等。
   - `additional_tokens_list`（6 个）：`<tool_call>`、`</tool_call>`、`<tool_response>`、`</tool_response>`、`<think>`、`</think>`。
     - **注意第 81–84 行的细节**：MiniMind 把这 6 个标签在 `tokenizer.json` 中的 `"special"` 设为了 `False`！为什么要这样做？因为如果设为 `special: True`，调用 `tokenizer.decode(..., skip_special_tokens=True)` 时就会把 `<think>` 和 `<tool_call>` 标签静默吞掉，导致外部解析器看不到思考过程和工具调用！
   - `buffer_tokens`（9 个）：`<|buffer1|>` ~ `<|buffer9|>`，凑齐 36 个特殊槽位，方便以后扩展新控制符而不改变普通词的 Token ID。

2. **中文流式解码的「半个汉字」问题（第 175–185 行）**：
   一个中文字符在 UTF-8 下占 **3 个字节**。在词表只有 6400 时，常见汉字会被合并成一个完整 token，但低频汉字可能被拆成 2 个 token（比如前 2 字节是一个 token，第 3 字节是另一个 token）。
   当模型逐 token 流式输出时，如果刚吐出前 2 个字节就立刻 `decode`，不完整的 UTF-8 字节序列会解码出替换字符 ``（Unicode `\ufffd`）。
   所以 MiniMind 在 [train_tokenizer.py#L177-L185](../minimind/trainer/train_tokenizer.py#L177-L185) 展示了标准的**流式字节缓冲写法**：
   ```python
   token_cache.append(tid)
   current_decode = tokenizer.decode(token_cache)
   if current_decode and '\ufffd' not in current_decode:
       print(current_decode)
       token_cache = []  # 只有凑齐完整 UTF-8 字符（不含 \ufffd）时才输出并清空缓冲！
   ```

3. **Jinja2 `chat_template` 完全拆解（第 121 行）**：
   MiniMind 采用了与 Qwen3 对齐的 ChatML 格式。一段多轮对话经 `tokenizer.apply_chat_template(messages, tokenize=False)` 渲染后长这样：
   ```text
   <|im_start|>system
   你是一个知识丰富的AI。<|im_end|>
   <|im_start|>user
   你好<|im_end|>
   <|im_start|>assistant
   <think>
   用户向我打招呼，我应该礼貌回应。
   </think>

   你好！有什么我可以帮你的吗？<|im_end|>
   ```
   观察模板末尾（第 121 行最后几行）：
   ```jinja2
   {%- if add_generation_prompt %}
       {{- '<|im_start|>assistant\n' }}
       {%- if open_thinking is defined and open_thinking is true %}
           {{- '<think>\n' }}
       {%- else %}
           {{- '<think>\n\n</think>\n\n' }}
       {%- endif %}
   {%- endif %}
   ```
   **这就是 MiniMind「自适应思考（Adaptive Thinking）」的全部秘密**：
   - 当推理时不开启思考（`open_thinking=False`），模板直接在 `assistant\n` 后面替模型把空的 `<think>\n\n</think>\n\n` 填好，模型紧接着往后续写的就是正文回答！
   - 当开启思考（`open_thinking=True`），模板只填 `<think>\n`，引导模型先写出思考过程，再写 `</think>\n\n` 和正文回答！

---

## 🛠️ 本阶段动手实验（Labs）

1. 打开 [labs/basics.py](../labs/basics.py)，完成：
   - `TODO(ch01-1)`：`stable_softmax`（先减最大值再 `exp`）
   - `TODO(ch02-1)`：`cross_entropy`（用 `torch.logsumexp` 和 `gather`，过滤 `ignore_index`）
   - `TODO(ch02-2)`：`perplexity`
2. 打开 [labs/tokenizer_bpe.py](../labs/tokenizer_bpe.py)，完成：
   - `TODO(ch03-1)`：`get_stats`（统计相邻 token 对频次）
   - `TODO(ch03-2)`：`merge`（合并指定 token 对）
   - `TODO(ch03-3)`：`ByteLevelBPE.train`（BPE 训练循环）
   - `TODO(ch03-4)`：`ByteLevelBPE._encode_chunk`（按合并优先级编码）
3. 运行闯关测试验证你的实现：
   ```bash
   python tools/grade.py ch01 -v
   python tools/grade.py ch03 -v
   ```
