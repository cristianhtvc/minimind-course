# 第二阶段上（第 4–7 章）：现代 Transformer 核心组件逐个击破

> **本阶段目标**：吃透 MiniMind（以及 LLaMA-3 / Qwen3）与 2017 年原版《Attention Is All You Need》相比的 **四大架构演进**：
> 1. **LayerNorm $\to$ Pre-Norm RMSNorm**（第 4 章）
> 2. **MHA $\to$ GQA + QK-Norm + KV Cache + FlashAttention**（第 5 章）
> 3. **Sinusoidal 绝对位置编码 $\to$ RoPE 旋转位置编码 + YaRN 长文本外推**（第 6 章）
> 4. **ReLU FFN $\to$ SwiGLU 门控前馈网络**（第 7 章）
>
> **配套代码与练习**：
> - 参考实现：[tinymind/norm.py](../tinymind/norm.py)、[tinymind/attention.py](../tinymind/attention.py)、[tinymind/rope.py](../tinymind/rope.py)、[tinymind/ffn.py](../tinymind/ffn.py)
> - 动手练习：[labs/norm.py](../labs/norm.py)、[labs/attention.py](../labs/attention.py)、[labs/rope.py](../labs/rope.py)、[labs/ffn.py](../labs/ffn.py)
> - 对照 MiniMind 源码：[minimind/model/model_minimind.py](../minimind/model/model_minimind.py)（第 50–147 行）
> - 闯关评测：`python tools/grade.py ch04`、`ch05`、`ch06`、`ch07`

---

## 第 4 章：RMSNorm 与 Pre-Norm 残差结构

### 4.1 为什么现代 LLM 都用 RMSNorm 替代了 LayerNorm？

对于一个维度为 $d$ 的隐藏向量 $\mathbf{x} = (x_1, \dots, x_d)$：
- **经典 LayerNorm**：先减均值 $\mu = \frac{1}{d}\sum_i x_i$，再除以标准差，最后乘 $\gamma$ 加 $\beta$：
  $$\text{LN}(\mathbf{x})_i = \frac{x_i - \mu}{\sqrt{\frac{1}{d}\sum_{j=1}^d (x_j - \mu)^2 + \epsilon}} \cdot \gamma_i + \beta_i$$
- **RMSNorm（Root Mean Square Layer Normalization，Zhang & Sennrich 2019）**：假设均值中心化（$-\mu$）对成功并不关键，直接用**均方根（RMS）**做缩放，并去掉偏置 $\beta$：
  $$\text{RMSNorm}(\mathbf{x})_i = \frac{x_i}{\sqrt{\frac{1}{d}\sum_{j=1}^d x_j^2 + \epsilon}} \cdot \gamma_i$$

**好处**：少算一次均值（省去一次向量规约通信/访存），参数量减半（没有 $\beta$），在实际 GPU 上比 LayerNorm 快 $10\% \sim 30\%$，收敛效果完全持平。

### 4.2 对照 MiniMind 源码：`RMSNorm` 的精度细节

查看 [minimind/model/model_minimind.py#L50-L60](../minimind/model/model_minimind.py#L50-L60)：
```python
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return (self.weight * self.norm(x.float())).type_as(x)
```
**为什么要写 `self.norm(x.float())` 然后再 `.type_as(x)`？**
因为在混合精度训练（`float16` 或 `bfloat16`）下，`x.pow(2)` 很容易超出 `float16` 的最大表示范围（$65504$），或者在累加求均值时损失尾数精度。先提升到 `float32` 算平方均值和倒数平方根 `torch.rsqrt`，再转回输入类型，是所有大模型（LLaMA/Qwen/MiniMind）的标准防溢出写法。

### 4.3 Post-Norm vs Pre-Norm

- **2017 原版 Transformer（Post-Norm）**：$\mathbf{x}_{l+1} = \text{Norm}(\mathbf{x}_l + F_l(\mathbf{x}_l))$。主干残差通路被夹在 `Norm` 里面，深层网络反向传播时梯度每过一层都要被 `Norm` 的雅可比矩阵缩放一次，极易梯度消失，必须靠漫长的 Warmup 才能训动。
- **MiniMind / 现代 LLM（Pre-Norm）**：
  $$\mathbf{h} = \mathbf{x}_l + \text{Attn}(\text{RMSNorm}(\mathbf{x}_l)), \qquad \mathbf{x}_{l+1} = \mathbf{h} + \text{FFN}(\text{RMSNorm}(\mathbf{h}))$$
  残差主干是一条从最后一层直通第一层的「梯度高速公路」（$\frac{\partial \mathbf{x}_L}{\partial \mathbf{x}_0} = \mathbf{I} + \dots$），训练极其稳定。

---

## 第 5 章：GQA 分组查询注意力、QK-Norm 与 KV Cache

### 5.1 从 MHA 到 MQA 再到 GQA（Grouped Query Attention）

在自回归推理生成第 $t$ 个 token 时，瓶颈不是算力，而是**把前 $t-1$ 个 token 缓存的 Key 和 Value（即 KV Cache）从显存搬运到 GPU 计算核心的带宽**。

| 注意力变体 | Query 头数 | Key/Value 头数 | 特点与权衡 |
| :--- | :---: | :---: | :--- |
| **MHA**（多头注意力） | $H_q$ | $H_{kv} = H_q$ | 表达力最强，但 KV Cache 显存占用最大 |
| **MQA**（多查询注意力） | $H_q$ | $H_{kv} = 1$ | 所有 Q 头共享唯一 1 组 KV 头，KV Cache 省 $H_q$ 倍，但质量有损 |
| **GQA**（分组查询注意力） | $H_q$ | $1 < H_{kv} < H_q$（整除） | 把 Q 头分成 $H_{kv}$ 组，每组 $n_{\text{rep}} = H_q / H_{kv}$ 个 Q 头共享 1 个 KV 头 |

MiniMind 默认配置：`num_attention_heads = 8`，`num_key_value_heads = 4`，`head_dim = 768 // 8 = 96`。即每 $n_{\text{rep}} = 2$ 个 Query 头共享 1 个 KV 头，**KV Cache 显存直接减半**！

#### `repeat_kv` 的零拷贝广播技巧
查看 [minimind/model/model_minimind.py#L86-L89](../minimind/model/model_minimind.py#L86-L89)：
```python
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1: return x
    return (x[:, :, :, None, :]
            .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
            .reshape(bs, slen, num_key_value_heads * n_rep, head_dim))
```
先在 `num_key_value_heads` 后面插入一个长度为 1 的新维度 `[:, :, :, None, :]`，用 `.expand(...)` 在视图上广播成 `n_rep` 份（不分配新内存），再 `.reshape(...)` 合并成 `num_key_value_heads * n_rep` 个头。

### 5.2 Qwen3 风格的 QK-Norm

查看 [minimind/model/model_minimind.py#L104-L105, L117](../minimind/model/model_minimind.py#L104-L105)：
```python
self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
...
xq, xk = self.q_norm(xq), self.k_norm(xk)
```
这是 MiniMind-3 对齐 **Qwen3** 的重要改进：在分完多头之后、施加 RoPE 之前，对每个头的 $q, k \in \mathbb{R}^{\text{head\_dim}}$ 各自做一次 `RMSNorm`。
**为什么需要 QK-Norm？** 随着训练步数增加或模型变深，$q$ 和 $k$ 的向量范数可能不断增长，导致点积 $q \cdot k^T / \sqrt{d_k}$ 变得极大，Softmax 进入饱和区（梯度趋近于 0），称为 **Attention Logit Growth / Entropy Collapse**。加上 QK-Norm 后，点积大小被严格约束住，可以放心使用更大的学习率而不崩。

### 5.3 KV Cache 与因果掩码切片细节

在自回归推理中，生成分为两个阶段：
1. **Prefill（预填充）阶段**：输入长度为 $L_0$ 的 Prompt，`past_key_value = None`。此时 `seq_len` $= L_0$，算出所有位置的 $K, V$ 并缓存下来：`past_kv = (xk, xv)`。
2. **Decode（逐词解码）阶段**：每次只输入刚生成的 **1 个新 token**（`seq_len = 1`），`past_key_value` 里存着前 $L_{\text{past}}$ 个 token 的 $(K_{\text{past}}, V_{\text{past}})$。
   - 拼接后的 `xk, xv` 长度为 $L_{\text{total}} = L_{\text{past}} + \text{seq\_len}$（[model_minimind.py#L120-L122](../minimind/model/model_minimind.py#L120-L122)）。
   - 注意此时 `xq` 的长度是 `seq_len`，而 `xk` 的长度是 $L_{\text{total}}$，因此注意力分数矩阵 `scores = xq @ xk.transpose(-2, -1)` 的形状是 `[bsz, heads, seq_len, L_total]`！

现在看 [model_minimind.py#L129](../minimind/model/model_minimind.py#L129) 这一行极其精妙的代码：
```python
if self.is_causal:
    scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
```
- **为什么是 `scores[:, :, :, -seq_len:]`**？
  因为 `scores` 的前 $L_{\text{past}}$ 列对应的是历史已经生成的 token，当前新输入的 `seq_len` 个 token **完全有权看到所有历史 token**（那部分掩码全为 0）；只有最后 `seq_len` 列（当前输入的 token 互相之间）需要加上上三角 $-\infty$ 因果掩码！
  - 特例：当 Decode 阶段 `seq_len = 1` 时，`1x1` 矩阵的 `.triu(1)` 全是 `0`，即唯一的那个新 token 可以看到包括自己在内的全部 $L_{\text{total}}$ 个位置！

---

## 第 6 章：RoPE 旋转位置编码与 YaRN 长上下文外推

### 6.1 RoPE 的数学直觉：为什么旋转能编码相对位置？

我们希望找一个位置变换 $f(\mathbf{v}, m)$，使得位置 $m$ 的 Query 与位置 $n$ 的 Key 的内积**只取决于它们之间的相对距离 $(m - n)$**：

$$\langle f(\mathbf{q}, m),\, f(\mathbf{k}, n) \rangle = g(\mathbf{q}, \mathbf{k}, m - n)$$

Su et al. (2021) 提出的 **RoPE（Rotary Position Embedding）** 给出了一个极其优美的几何解答：
把 $d$ 维向量拆成 $d/2$ 对二维向量 $(x^{(1)}, x^{(2)})$，看作复平面上的复数 $x^{(1)} + i x^{(2)}$。在位置 $m$ 上，把它**逆时针旋转角度 $m\theta$**：

$$\begin{pmatrix} \tilde{x}^{(1)}_m \\ \tilde{x}^{(2)}_m \end{pmatrix} = \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix} \begin{pmatrix} x^{(1)} \\ x^{(2)} \end{pmatrix} = \begin{pmatrix} x^{(1)} \cos m\theta - x^{(2)} \sin m\theta \\ x^{(2)} \cos m\theta + x^{(1)} \sin m\theta \end{pmatrix}$$

由于两个向量在平面上同时旋转相同角度时内积不变，位置 $m$ 的 $\tilde{\mathbf{q}}_m$（转了 $m\theta$）与位置 $n$ 的 $\tilde{\mathbf{k}}_n$（转了 $n\theta$）的内积，只取决于它们的**夹角之差 $(m-n)\theta$**！

不同维度对 $i \in \{0, 1, \dots, d/2 - 1\}$ 使用从快到慢的几何级数频率（就像手表的秒针、分针、时针）：

$$\theta_i = \frac{1}{\text{base}^{2i / d}}, \qquad \text{MiniMind 中 } \text{base} = \texttt{rope\_theta} = 10^6$$

### 6.2 `rotate_half` 分半配对写法

观察上面的二维旋转公式右端：
$$\begin{pmatrix} x^{(1)} \\ x^{(2)} \end{pmatrix} \odot \begin{pmatrix} \cos m\theta \\ \cos m\theta \end{pmatrix} + \begin{pmatrix} -x^{(2)} \\ x^{(1)} \end{pmatrix} \odot \begin{pmatrix} \sin m\theta \\ \sin m\theta \end{pmatrix}$$

MiniMind（与 HuggingFace LLaMA/Qwen 一样）把向量的**前半段** $x_{0 \dots d/2-1}$ 和**后半段** $x_{d/2 \dots d-1}$ 配成 $d/2$ 对，定义：
```python
def rotate_half(x):
    return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
```
再把 `freqs_cos` 和 `freqs_sin` 前后复制拼接成 `[end, dim]`，就可以用一行向量化代码完成所有维度的旋转（[model_minimind.py#L80-L84](../minimind/model/model_minimind.py#L80-L84)）：
```python
q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
```

### 6.3 YaRN 长上下文外推（[model_minimind.py#L64-L73](../minimind/model/model_minimind.py#L64-L73)）

如果模型预训练时只见过长度 $L_{\text{orig}} = 2048$ 的序列，推理时直接喂入 $32768$（16 倍长度）的文本会发生什么？
- **直接外推**：低频维度（时针）在训练时只转了不到半圈，推理时突然转到从未见过的角度区间，注意力立刻崩溃。
- **线性位置插值（Position Interpolation）**：把所有频率都除以缩放倍数 $s = 16$（假装位置 $16$ 是原来的位置 $1$）。缺点是**高频维度（秒针）被挤压得太密**，模型分不清相邻几个 token 的先后顺序，局部理解能力下降。
- **YaRN（Yet another RoPE extensioN，Peng et al. 2023）的高低频分治思想**：
  计算每个维度 $i$ 在原始训练长度 $L_{\text{orig}}$ 内转过的圈数 $r(i) = \frac{L_{\text{orig}}}{2\pi / \theta_i}$：
  1. **高频维度**（圈数 $> \beta_{\text{fast}} = 32$）：训练时已经转了很多圈，见过所有角度，**完全不缩放**（$\gamma = 0$）。
  2. **低频维度**（圈数 $< \beta_{\text{slow}} = 1$）：训练时连一圈都没转完，**做完整线性插值**，频率除以 $s$（$\gamma = 1$）。
  3. **中频维度**：在 $\gamma \in [0, 1]$ 之间用线性斜坡 `ramp` 平滑过渡：
     $$\theta'_i = \theta_i \cdot \left( (1 - \gamma_i) + \frac{\gamma_i}{s} \right)$$
  这就是 [model_minimind.py#L70-L73](../minimind/model/model_minimind.py#L70-L73) 那 4 行浓缩代码的全部数学含义！

---

## 第 7 章：SwiGLU 前馈网络（`FeedForward`）

### 7.1 从两层 ReLU FFN 到三矩阵 SwiGLU

- **原版 Transformer FFN**：两个线性层，中间一个 ReLU：
  $$\text{FFN}(\mathbf{x}) = W_2 \, \text{ReLU}(W_1 \mathbf{x})$$
- **SwiGLU（Shazeer 2020，LLaMA/Qwen/MiniMind 标配）**：用三个无偏置线性矩阵 `gate_proj` ($W_g$)、`up_proj` ($W_u$)、`down_proj` ($W_d$)，其中一路过 $\text{SiLU}(z) = z \cdot \sigma(z)$ 作为**门控（Gate）**，与另一路逐元素相乘后再降维：

$$\text{SwiGLU}(\mathbf{x}) = W_d \Big( \text{SiLU}(W_g \mathbf{x}) \odot (W_u \mathbf{x}) \Big)$$

对应 [model_minimind.py#L145-L146](../minimind/model/model_minimind.py#L145-L146) 只有一行：
```python
return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

### 7.2 为什么 `intermediate_size = math.ceil(hidden_size * math.pi / 64) * 64`？

在原版两矩阵 FFN 中，中间层维度通常是 $4d$（参数量 $2 \times d \times 4d = 8d^2$）。换成三矩阵 SwiGLU 后，为了保持单层参数量近似不变，通常取 $\frac{8}{3}d \approx 2.67d$ 或 $\pi d \approx 3.14d$。
MiniMind 在 [model_minimind.py#L26](../minimind/model/model_minimind.py#L26) 取 $\lceil 768 \times \pi / 64 \rceil \times 64 = 2432$。**向上取整到 64 的整数倍**是为了对齐 GPU Tensor Core 的矩阵块（tile size），保证 GEMM 算子跑满硬件吞吐！

---

## 🛠️ 本阶段动手实验（Labs）

依次完成以下 4 个文件中的 TODO 并运行测试（每一项测试都会与 MiniMind 原版模块做**逐元素数值比对**）：

```bash
# 1. 完成 labs/norm.py 中的 TODO(ch04-1)
python tools/grade.py ch04 -v

# 2. 完成 labs/rope.py 中的 TODO(ch06-1) ~ TODO(ch06-5)
python tools/grade.py ch06 -v

# 3. 完成 labs/attention.py 中的 TODO(ch05-1) ~ TODO(ch05-4)
python tools/grade.py ch05 -v

# 4. 完成 labs/ffn.py 中的 TODO(ch07-1)
python tools/grade.py ch07 -v
```
