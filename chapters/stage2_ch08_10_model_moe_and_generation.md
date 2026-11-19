# 第二阶段下（第 8–10 章）：完整模型组装、混合专家（MoE）与自回归采样生成

> **本阶段目标**：把第 4–7 章的积木拼成完整的 `MiniMindForCausalLM`（Dense 64M 与 MoE 198M-A64M），学会逐层精算一个 LLM 的参数量，吃透 MoE 路由中的 Straight-Through Estimator（直通梯度估计）、负载均衡辅助损失 `aux_loss`，以及 `generate()` 自回归采样中的五大解码策略。
>
> **配套代码与练习**：
> - 参考实现：[tinymind/model.py](../tinymind/model.py)、[tinymind/moe.py](../tinymind/moe.py)、[tinymind/generate.py](../tinymind/generate.py)
> - 动手练习：[labs/model.py](../labs/model.py)、[labs/moe.py](../labs/moe.py)、[labs/generate.py](../labs/generate.py)
> - 对照 MiniMind 源码：[minimind/model/model_minimind.py](../minimind/model/model_minimind.py)（第 148–293 行）
> - 闯关评测：`python tools/grade.py ch08`、`python tools/grade.py ch10`

---

## 第 8 章：组装 `MiniMindForCausalLM` 与 64M 参数逐层精算

### 8.1 三层类继承体系与 HuggingFace 生态兼容

在 [minimind/model/model_minimind.py#L183-L258](../minimind/model/model_minimind.py#L183-L258) 中，模型由内到外分为三层：
1. **`MiniMindBlock`（单层 Transformer 解码器块）**：
   ```python
   residual = hidden_states
   hidden_states, present_key_value = self.self_attn(
       self.input_layernorm(hidden_states), position_embeddings,
       past_key_value, use_cache, attention_mask
   )
   hidden_states += residual
   hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
   ```
2. **`MiniMindModel`（主干网络）**：
   `embed_tokens` $\to$ `8 × MiniMindBlock` $\to$ 最终 `RMSNorm`，同时用 `register_buffer(..., persistent=False)` 持有预计算好的 `freqs_cos / freqs_sin`，并在最后汇总所有 MoE 层的 `aux_loss`。
   - **细节（第 218、224 行）**：当 `past_key_values` 不为空时，已缓存的序列长度为 `start_pos = past_key_values[0][0].shape[1]`，因此当前输入的 `seq_length` 个 token 对应的位置区间是 `[start_pos : start_pos + seq_length]`！
3. **`MiniMindForCausalLM`（带语言模型头的因果语言模型）**：
   在 `MiniMindModel` 之上加一个线性分类头 `lm_head = nn.Linear(hidden_size, vocab_size, bias=False)`，并计算错位交叉熵损失。

### 8.2 三个核心工程细节

1. **权重共享（Weight Tying，第 241、247 行）**：
   ```python
   if self.config.tie_word_embeddings:
       self.model.embed_tokens.weight = self.lm_head.weight
   ```
   输入词嵌入矩阵（把 token ID 查表变成 768 维向量）和输出投影矩阵（把 768 维向量点乘出 6400 个词的打分）在语义上都是「每个词的向量表示」。让它们指向**同一块内存**，不仅直接省下 $6400 \times 768 \approx 4.92\text{M}$ 参数，还能让低频词在输入和输出两端同时得到梯度更新。
2. **`logits_to_keep` 显存优化（第 252–253 行）**：
   ```python
   slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
   logits = self.lm_head(hidden_states[:, slice_indices, :])
   ```
   在推理或 RL 采样阶段，如果只需要最后 $K$ 个位置的 logits，就没必要把前面所有 Prompt 位置的隐藏状态都乘以 `lm_head`（形状 `[B, L, 6400]` 会白白占用显存）。`logits_to_keep=0` 时 `slice(0, None)` 取全部位置；`logits_to_keep=1` 时 `slice(-1, None)` 只算最后一个位置。
3. **错位对齐（Shift Logits & Labels，第 256 行）**：
   位置 $0 \dots L-2$ 的输出 `logits[..., :-1, :]` 用来预测位置 $1 \dots L-1$ 的目标 `labels[..., 1:]`。

### 8.3 手算 `minimind-3`（Dense 64M）的每一个参数！

默认配置：$V=6400,\; H=768,\; L=8,\; H_q=8,\; H_{kv}=4,\; d_h=96,\; I=2432$。

| 组件 | 形状 / 计算公式 | 单层参数量 | 占总模型层数 | 小计 |
| :--- | :--- | :---: | :---: | :---: |
| **词嵌入 `embed_tokens`**（与 `lm_head` 共享） | $V \times H = 6400 \times 768$ | — | $\times 1$ | $4,915,200$ |
| **Attention `q_proj` + `o_proj`** | $2 \times (H \times H_q d_h) = 2 \times 768 \times 768$ | $1,179,648$ | $\times 8$ | $9,437,184$ |
| **Attention `k_proj` + `v_proj`** (GQA!) | $2 \times (H \times H_{kv} d_h) = 2 \times 768 \times 384$ | $589,824$ | $\times 8$ | $4,718,592$ |
| **Attention `q_norm` + `k_norm`** | $2 \times d_h = 2 \times 96$ | $192$ | $\times 8$ | $1,536$ |
| **SwiGLU FFN (`gate` + `up` + `down`)** | $3 \times (H \times I) = 3 \times 768 \times 2432$ | $5,603,328$ | $\times 8$ | $44,826,624$ |
| **Block 内 2 个 `RMSNorm`** | $2 \times H = 2 \times 768$ | $1,536$ | $\times 8$ | $12,288$ |
| **最终 `model.norm`** | $H = 768$ | — | $\times 1$ | $768$ |
| **总计** | | **每层 $7,374,528$** | | **$63,912,192 \approx 63.91\text{M}$** |

> **思考题**：为什么 FFN 占了整层参数的 $5.60 / 7.37 \approx 76\%$？这也是为什么第 9 章的 MoE **只把 FFN 替换成多专家**，而不动 Attention！

---

## 第 9 章：混合专家模型 MoE（`MOEFeedForward`）

### 9.1 稀疏激活：198M 总参数 vs 64M 激活参数

在 `minimind-3-moe` 中，`use_moe = True`，默认 `num_experts = 4, num_experts_per_tok = 1`：
- 把每层的 1 个 `FeedForward`（5.60M 参数）换成 **4 个并行的 Expert `FeedForward`**（$4 \times 5.60\text{M} = 22.41\text{M}$ 参数）加 1 个极小的路由器 `gate = nn.Linear(768, 4)`。
- 8 层一共多了 $8 \times 3 \times 5.60\text{M} \approx 134.5\text{M}$ 参数，因此**总参数量变成 $63.91 + 134.5 \approx 198.4\text{M}$**。
- 但是每个 token 经过路由器打分后，**只选 Top-1 专家计算**，所以单次前向实际参与矩阵乘法的**激活参数量（Active Params）依然是 63.9M**（记作 `198M-A64M`）！

### 9.2 逐行精读 `MOEFeedForward` 的三个黑科技（[model_minimind.py#L148-L181](../minimind/model/model_minimind.py#L148-L181)）

#### 黑科技 1：当 $k=1$ 时的 Straight-Through Estimator（第 161–166 行）
```python
if self.config.norm_topk_prob:
    if self.config.num_experts_per_tok > 1:
        topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
    else:
        top1 = torch.topk(F.softmax(self.gate(x_flat.detach()), dim=-1), k=1, dim=-1, sorted=False)[0]
        topk_weight = top1 - top1.detach() + 1.0 # k=1: 1.0 forward on purpose, gradient via straight-through
```
**仔细看第 165–166 行！为什么不直接写 `topk_weight = 1.0`？**
- 当只选 $k=1$ 个专家且开启 `norm_topk_prob`（归一化选中专家的权重使其和为 1）时，唯一一个专家归一化后的权重在数值上就等于常数 $1.0$。
- 但如果直接把 `topk_weight` 写成常数 `1.0`，专家输出 `expert(x) * 1.0` 就**跟路由器 `self.gate` 完全断开计算图了**！路由器 `self.gate` 将收不到任何来自主任务交叉熵损失的梯度！
- MiniMind 用了 **STE（直通估计器）** 技巧：`top1 - top1.detach() + 1.0`：
  - **前向传播时**：`top1` 与 `top1.detach()` 数值相等，相减为 0，结果恒等于 `1.0`；
  - **反向传播时**：`.detach()` 和 `+ 1.0` 的导数为 0，梯度直接从 `top1` 原封不动地传回 `self.gate`！

#### 黑科技 2：防止 DDP 多卡死锁的「零乘哑梯度」（第 174–175 行）
```python
elif self.training:
    y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
```
在某个小 batch 里，如果某个专家 `i` 运气不好，没有任何一个 token 选它（`mask.any() == False`），那么该专家的参数在这个 step 就不会参与前向计算。在 PyTorch `DistributedDataParallel (DDP)` 多卡训练时，没参与计算的参数收不到梯度，会导致 All-Reduce 通信挂起或报错！
加上 `0 * sum(p.sum() ...)` 后，数值上加的是 0，但所有专家参数都被挂进了计算图（梯度为 0），DDP 就能安全同步。

#### 黑科技 3：Switch Transformer 负载均衡辅助损失 `aux_loss`（第 176–178 行）
如果没有约束，路由器很容易陷入「赢家通吃」的坍塌状态：一开始稍微好一点的专家被选得越来越多、训得越来越好，其余 3 个专家彻底饿死。
为此引入负载均衡损失（Switch Transformer, Fedus et al. 2021）：
- 设 $f_i$（代码中 `load`）为本 batch 中**实际被分给专家 $i$ 的 token 比例**：
  $$f_i = \frac{1}{T} \sum_{t=1}^T \mathbb{I}(\text{token } t \text{ 选中专家 } i)$$
- 设 $P_i$（代码中 `scores.mean(0)`）为本 batch 中**路由器给专家 $i$ 的平均路由概率**：
  $$P_i = \frac{1}{T} \sum_{t=1}^T p_i(x_t)$$
- 辅助损失为两者内积乘以专家数 $E$ 和系数 $\alpha$（默认 $5\times 10^{-4}$）：
  $$\mathcal{L}_{\text{aux}} = \alpha \cdot E \sum_{i=1}^E f_i \cdot P_i$$
  当且仅当所有专家均匀分配（$f_i = P_i = 1/E$）时，$\sum_{i=1}^E f_i P_i$ 取得最小值 $1/E$（此时 $E \sum f_i P_i = 1$）。

---

## 第 10 章：自回归解码与采样策略（`generate`）

训练好模型后，如何从 `logits`（形状 `[B, V]`）里挑出下一个 token？如果永远选概率最大的那个（Greedy Decoding），模型遇到开放式问题很容易陷入死循环复读。MiniMind 在 [model_minimind.py#L261-L293](../minimind/model/model_minimind.py#L261-L293) 用短短 30 行实现了完整的工业级采样流水线，顺序如下：

1. **温度缩放（Temperature，第 273 行）**：
   $$\tilde{z}_i = \frac{z_i}{\tau}$$
   - $\tau \to 0$：差距被无限放大，退化为 Greedy `argmax`（最确定、最保守）；
   - $\tau = 1$：原始分布；$\tau > 1$：分布变平坦，更有创造力但也更容易胡言乱语。
2. **重复惩罚（Repetition Penalty，第 274–276 行）**：
   对上下文 `input_ids[i]` 中**已经出现过**的 token `seen`，设惩罚系数 $r \ge 1.0$：
   $$\tilde{z}_v = \begin{cases} z_v / r & \text{若 } z_v > 0 \\ z_v \cdot r & \text{若 } z_v \le 0 \end{cases}$$
   **为什么要分正负？** 因为无论是正 logit 除以 $r$（变小）还是负 logit 乘以 $r$（变得更负），经过 Softmax 后的概率都会单调减小！
3. **Top-K 过滤（第 277–278 行）**：
   只保留 logit 最大的前 $K$ 个候选词，其余全部置为 $-\infty$（`softmax(-inf) = 0`），砍掉长尾的离谱噪音词。
4. **Top-P（核采样 Nucleus Sampling，第 279–282 行）**：
   把概率从大到小排序并求累积和 `cumsum`，只保留累积概率刚达到 $p$（如 $0.85$）的最小核心词集。
   - **细节（第 282 行）**：`F.pad(mask[..., :-1], (1, 0))` 把布尔掩码向右平移了一位，左侧补 `False`。为什么？因为如果第 1 个词的概率本身就是 $0.95 > p$，不右移的话连第 1 个词都会被 `mask` 成 `True`（置为 $-\infty$），导致所有词都是 $-\infty$，Softmax 变成 `NaN`！右移一位保证了**至少保留第一个超过阈值的词**。
5. **多项式采样与 Batch `finished` 状态机（第 283–290 行）**：
   `torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)` 按概率采样；在一个 batch 中，已经生成过 `eos_token_id` 的样本会被记录在布尔向量 `finished` 中，后续步骤强制将其新 token 设为 `eos_token_id`，直到 `finished.all()` 全 batch 结束。

---

## 🛠️ 本阶段动手实验（Labs）

```bash
# 1. 完成 labs/model.py 中的 TODO(ch08-1) ~ TODO(ch08-3)
#    以及 labs/moe.py 中的 TODO(ch09-1) ~ TODO(ch09-3)
python tools/grade.py ch08 -v

# 2. 完成 labs/generate.py 中的 TODO(ch10-1) ~ TODO(ch10-4)
python tools/grade.py ch10 -v
```
