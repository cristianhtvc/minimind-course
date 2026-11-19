# 第 0 篇（必读基石）：大模型四大底层技术栈与 22 篇奠基论文全景精读

> **为什么要先读这一篇？**
> 很多教程一上来就让你写 `x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)` 或 `xq.transpose(1, 2)`，却从不解释：
> - 为什么 `transpose` 之后调 `.view()` 会报错，必须先 `.contiguous()`？
> - 为什么 `RMSNorm` 和 `Attention` 里要先 `.float()` 再 `.type_as(x)`？不转为什么在 `float16` 下会直接炸出 `NaN`？
> - 为什么自回归生成第 1000 个词时，GPU 利用率看起来很高，算力（TFLOPS）却连 5% 都没发挥出来？GQA、KV Cache、FlashAttention 在硬件层面到底省了什么？
> - 为什么 MoE 里 `top1 - top1.detach() + 1.0` 这种「自己减自己」的代码不是废话，而是离散算子反向传播的核心技巧？
>
> 本篇将把 **PyTorch 底层内存与自动微分机制、IEEE-754 浮点精度、GPU 硬件体系结构**，以及支撑 MiniMind 全项目的 **22 篇核心学术论文** 一次性讲透。

---

## 第一部分：支撑 LLM 的四大底层技术栈基石

### 基石 1：PyTorch Tensor 的物理内存布局 —— `Storage`、`Stride` 与零拷贝视图

在计算机物理内存（RAM 或 GPU 显存）中，**根本不存在多维数组，所有数据都是一维连续线性排列的字节数组**（在 PyTorch 底层称为 `Storage`）。
一个形状为 `shape = (B, L, H, D)` 的四维张量之所以看起来像四维，是因为 Tensor 头信息（`TensorImpl`）里维护了一个叫做 **步幅（`stride`）** 的元组：

$$\text{stride} = (s_0, s_1, s_2, s_3)$$

它的物理含义是：**在逻辑维度 $k$ 上前进 1 个单位，在一维物理内存 `Storage` 中需要跳过 $s_k$ 个元素**。因此，逻辑坐标 $(i_0, i_1, i_2, i_3)$ 对应的物理内存下标为：

$$\text{storage\_index} = \text{storage\_offset} + i_0 s_0 + i_1 s_1 + i_2 s_2 + i_3 s_3$$

对于刚创建的「行优先连续（Row-Major Contiguous）」张量 `x`（形状 `(2, 4, 8, 96)`）：
- $s_3 = 1$（最后一维相邻元素在物理内存中紧挨着）
- $s_2 = 96$
- $s_1 = 8 \times 96 = 768$
- $s_0 = 4 \times 768 = 3072$

现在你就能彻底看懂 MiniMind 源码里的 4 个高频张量操作在底层干了什么：

| 操作 | MiniMind 典型代码位置 | 底层物理内存发生了什么？ | 是否拷贝显存？ |
| :--- | :--- | :--- | :---: |
| **`.view(...)`** | `xq.view(bsz, seq_len, n_heads, head_dim)` | 不动 `Storage`，只把原来的步幅 $s_1 = 768$ 拆成 $(s_1=768, s_2=96, s_3=1)$。**前提：张量在物理内存上必须是连续的（`is_contiguous() == True`）**。 | **零拷贝 $O(1)$** |
| **`.transpose(1, 2)`** | `xq.transpose(1, 2)`（交换 `seq_len` 与 `heads`） | 不动 `Storage`，只交换 `shape` 和 `stride` 的第 1、2 维：`stride` 变成 `(3072, 96, 768, 1)`。**注意：此时物理内存不再按新维度顺序连续（`is_contiguous() == False`）！** | **零拷贝 $O(1)$** |
| **`.contiguous()`** | `logits[..., :-1, :].contiguous()` | 当对切片或 `transpose` 后的非连续张量调用 `.view()` 时，由于步幅不满足连续条件，必须先调 `.contiguous()` 在显存中重新开辟一块按新逻辑顺序排列的物理内存（或者直接用 `.reshape()`，它会在非连续时自动触发 `.contiguous().view()`）。 | **发生显存重排拷贝** |
| **`.expand(...)`** | `repeat_kv` 中的 `x[:, :, :, None, :].expand(...)` | 把长度为 `1` 的新维度步幅设为 **`stride = 0`**！意思是无论该维度下标取 $0, 1, \dots, n_{\text{rep}}-1$，物理内存偏移量都加 $0$（读取同一份 KV 头数据）！然后再接 `.reshape(...)` 时才真正物化。 | **`expand` 阶段零拷贝** |

---

### 基石 2：IEEE-754 浮点数位级拆解 —— 为什么大模型偏爱 `BF16`？

计算机用有限的二进制位表示实数：$(-1)^{\text{sign}} \times 2^{\text{exponent} - \text{bias}} \times (1.\text{mantissa})_2$。
对比深度学习中的三种核心浮点格式：

```text
FP32 (32 bits): [ 1位符号 S |      8位指数 E (bias=127)      |            23位尾数 M (精度 ~7位十进制)            ]
FP16 (16 bits): [ 1位符号 S |  5位指数 E  |       10位尾数 M (精度 ~3.3位十进制)       ]  <-- 指数只有5位！最大正数仅 65504！
BF16 (16 bits): [ 1位符号 S |      8位指数 E (bias=127)      |   7位尾数 M (~2.4位)   ]  <-- 指数与 FP32 完全相同！
```

| 浮点格式 | 字节数 | 指数位 | 尾数位 | 最大可表示正数 | 最小正规正数 | 核心优缺点与 MiniMind 中的对策 |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **FP32** | 4 B | 8 | 23 | $\approx 3.40 \times 10^{38}$ | $\approx 1.18 \times 10^{-38}$ | 精度最高、绝不轻易溢出；用于存放优化器动量、`RMSNorm` 平方均值、Attention `scores` 与 Softmax |
| **FP16** | 2 B | **5** | 10 | **$65,504$** | $6.10 \times 10^{-5}$ | **动态范围极窄！** ① 前向时任何中间值 $> 65504$（例如 `x.pow(2)` 中只要某元素 $> 256$）立刻变 `inf` $\to$ `NaN`；② 反向传播时小于 $6 \times 10^{-8}$ 的微小梯度直接下溢变成 `0`。因此用 FP16 **必须配 `GradScaler`**（先把 Loss 乘以 $2^{16}$ 放大梯度，更新前再 `unscale_` 除回来） |
| **BF16** | 2 B | **8** | 7 | **$\approx 3.39 \times 10^{38}$** | **$\approx 1.18 \times 10^{-38}$** | 直接截断 FP32 的后 16 位尾数得到！拥有与 FP32 **完全相同的动态范围**，彻底告别 `65504` 上溢与梯度下溢，无需 `GradScaler`（MiniMind 中 `GradScaler(enabled=(args.dtype == 'float16'))` 在 BF16 下自动关闭） |

**这就是为什么 MiniMind 在 `model_minimind.py` 中有两处看似多余、实则救命的 `.float()`**：
1. **`RMSNorm.forward`（第 60 行）**：`self.norm(x.float()).type_as(x)` —— 如果不在 FP32 下算 `x.pow(2).mean(-1)`，FP16 下激活值超过 $256$ 平方即溢出，BF16 下 7 位尾数在几百个维度累加求平均时也会产生显著舍入误差。
2. **`Attention.forward`（第 128 行）**：`scores = (xq @ xk.transpose(-2, -1)).float() / math.sqrt(self.head_dim)` —— Softmax 包含指数运算 $e^{s_i}$，必须在 FP32 下完成点积缩放与 Softmax，再 `.type_as(xq)` 转回半精度乘以 $V$。

---

### 基石 3：GPU 硬件存储层次与 Roofline 模型 —— 为什么推理需要 KV Cache、GQA 和 FlashAttention？

一块现代 GPU（以 NVIDIA A100 或 RTX 3090 为例）内部有两个截然不同的世界：
1. **计算核心（Tensor Cores / CUDA Cores）**：算力极强（A100 BF16 算力高达 **$312\text{ TFLOPS}$**，即每秒 $3.12 \times 10^{14}$ 次浮点运算）。
2. **两级存储层次**：
   - **片上高速缓存 SRAM（寄存器 + L1/Shared Memory）**：速度极快（带宽 **$\sim 19\text{ TB/s}$**），但容量极小（仅 **$20 \sim 40\text{ MB}$**）。
   - **片外高带宽显存 HBM / GDDR6X（平时说的 24GB/80GB 显存）**：容量大，但带宽只有 **$1.0 \sim 2.0\text{ TB/s}$**（比算力和 SRAM 慢一个数量级！）。

定义一个算子的 **算术强度（Arithmetic Intensity）**：

$$\mathcal{I} = \frac{\text{浮点运算次数 (FLOPs)}}{\text{从 HBM 显存读写的数据字节数 (Bytes)}}$$

- 当 $\mathcal{I} > \text{硬件拐点}$（A100 上约为 $312\text{ TFLOPS} / 1.5\text{ TB/s} \approx \mathbf{200\text{ FLOPs/Byte}}$）时，算子是 **Compute-Bound（算力受限）**，Tensor Core 满载运行；
- 当 $\mathcal{I} < 200$ 时，算子是 **Memory-Bandwidth-Bound（显存带宽受限）**：**计算核心绝大部分时间都在干等数据从 HBM 显存搬运到 SRAM！**

#### 用 Roofline 模型一眼看穿 LLM 的四大架构设计：

1. **为什么自回归生成（Decode）必须用 KV Cache，且用了之后是极端显存带宽受限？**
   - 假设已生成 $L$ 个 token，现在生成第 $L+1$ 个 token：如果不缓存历史每层的 $\mathbf{k}_{\le L}, \mathbf{v}_{\le L}$，每生成一个新词都要把前 $L$ 个词重新过一遍 8 层 Transformer，生成长度为 $N$ 的回答总计算量高达 $O(N^2 d^2 + N^3 d)$！
   - 加上 **KV Cache**（`past_key_value`）后，历史 $\mathbf{K}, \mathbf{V}$ 直接从显存读取，每次只输入 **1 个新 token**（矩阵乘矩阵 $M \times K \times N$ 退化为 **向量乘矩阵 $1 \times K \times N$**）。此时每从显存搬运一个权重参数 $w_{ij}$（2 字节），只做 2 次浮点运算（1 次乘 1 次加），算术强度 $\mathcal{I} \approx \mathbf{1\text{ FLOP/Byte}} \ll 200$！因此 **Decode 阶段的速度 100% 取决于你要从 HBM 显存搬运多少字节的权重和 KV Cache**！
2. **为什么 GQA（Grouped Query Attention）能大幅加速推理并省显存？**
   因为 Decode 阶段每步都要把全部历史 KV Cache 从 HBM 搬进 SRAM。MiniMind 用 `num_key_value_heads = 4` 替代 `num_attention_heads = 8`，**KV Cache 的显存占用和每步 HBM 搬运字节数直接砍半**！
3. **为什么 FlashAttention（`F.scaled_dot_product_attention`）比手写 `softmax(Q @ K.T) @ V` 快几倍还省显存？**
   - 手写注意力在序列长度 $L = 8192$ 时，必须先把完整的 $L \times L = 67,108,864$ 个元素的注意力矩阵 $\mathbf{S} = \mathbf{Q}\mathbf{K}^T$ 写回慢速 HBM 显存，再从 HBM 读出来算 Softmax $\mathbf{P}$ 写回 HBM，再从 HBM 读出来乘以 $\mathbf{V}$——**在 HBM 上反复读写巨大的 $O(L^2)$ 中间矩阵**！
   - **FlashAttention（Dao et al. 2022）** 利用 **Tiling（分块）+ Online Softmax**（在遍历分块时动态更新最大值 $m$ 和分母局部和 $e^{m_{\text{old}} - m_{\text{new}}} \ell_{\text{old}}$），**全程在片上高速 SRAM 内一口气算完 $\text{softmax}(\mathbf{Q}\mathbf{K}^T)\mathbf{V}$，完全不在 HBM 中实例化 $L \times L$ 的注意力矩阵**！显存占用从 $O(L^2)$ 降为 $O(L)$。
4. **为什么 MiniMind 的 `intermediate_size = ceil(768 * pi / 64) * 64 = 2432` 要对齐 64 的倍数？**
   GPU 的 Tensor Core 是以 $16 \times 16$ 或 $64 \times 64$ 的矩阵块（Tile）为单位做硬件级 MMA（Matrix Multiply-Accumulate）指令的，且 HBM 显存事务以 128 字节（恰好 64 个 FP16/BF16 元素）对齐。如果维度不是 64 的倍数，边界 Tile 就必须补零并触发非对齐访存，白白浪费高达 $20\%$ 的硬件吞吐。

---

### 基石 4：PyTorch Autograd 动态计算图与直通梯度估计（STE）

当你在 PyTorch 中写下 `y = a * x + b` 时，如果 `x.requires_grad == True`，PyTorch 不仅算出数值 `y`，还会在后台 C++ 引擎中创建一个 `AutogradMeta` 节点，把 `y.grad_fn` 指向 `<AddBackward0>`，再指向 `<MulBackward0>`，构成一张**有向无环计算图（DAG）**。调用 `loss.backward()` 时，引擎按拓扑逆序遍历这张图，自动执行链式法则（向量-雅可比积 VJP）。

#### 1. `detach()` vs `torch.no_grad()` vs `@torch.inference_mode()`
- **`x.detach()`**：创建一个与 `x` **共享底层物理内存 `Storage`** 的新 Tensor 视图，但将其从当前计算图中剪断（`requires_grad=False, grad_fn=None`）。梯度传到 `x.detach()` 就会停住。
- **`with torch.no_grad():`**（MiniMind 用于 DPO 的 `ref_model`、蒸馏的 `teacher_model`、PPO 的 Rollout 阶段）：在该代码块内产生的所有新张量都不构建计算图节点，**省去保存前向中间激活值的显存**。
- **`@torch.inference_mode()`**（MiniMind 用于 `MiniMindForCausalLM.generate` 第 261 行）：比 `no_grad()` 更彻底的极致推理模式——不仅不建计算图，连 Tensor 的版本计数器（`version_counter`）和视图追踪都全部关闭，C++ 调度开销最低。

#### 2. 直通梯度估计器（Straight-Through Estimator, STE）
在深度学习中，一旦遇到**离散不可导算子**（例如阶跃函数、取整 $\text{round}(x)$，或 MiniMind MoE 中 $k=1$ 时把 Top-1 专家权重归一化为常数 $1.0$），其数学导数处处为 $0$，反向传播时梯度在这一层就会彻底断掉。
**STE 的核心魔术**是利用 `.detach()` 构造一个**前向值与反向导数解耦**的表达式：

$$w_{\text{STE}} = \underbrace{p_{\text{top1}}}_{\text{有梯度}} - \underbrace{\text{sg}(p_{\text{top1}})}_{\text{无梯度，数值与左项相等}} + 1.0$$

在 PyTorch 中写作（见 [model_minimind.py#L166](../minimind/model/model_minimind.py#L166)）：
```python
topk_weight = top1 - top1.detach() + 1.0
```
- **前向求值（Forward）**：`top1` 与 `top1.detach()` 数值一模一样，相减等于 $0$，所以 `topk_weight` 的数值严格等于 **$1.0$**！
- **反向求导（Backward）**：常数 `1.0` 和被截断的 `top1.detach()` 导数都是 $0$，因此：
  $$\frac{\partial w_{\text{STE}}}{\partial \text{top1}} = 1 - 0 + 0 = 1$$
  来自专家输出的上游梯度原封不动地穿过 `topk_weight` 流回路由器 `self.gate`！

---

## 第二部分：支撑 MiniMind 的 22 篇奠基与前沿论文精读图谱

下面按课程的 6 大阶段，梳理你在学习每一章时应当对照了解的经典与前沿论文。

### 阶段一：语言模型与分词器（第 1–3 章）

| 序号 | 论文标题、作者与发表信息 | 解决的核心痛点 | 论文核心思想与公式 | MiniMind 对应实现 |
| :---: | :--- | :--- | :--- | :--- |
| **P01** | **A Neural Probabilistic Language Model**<br>Bengio et al., *JMLR 2003* | 传统 N-gram 语言模型遇到未见过的词组合时概率为 0（维度灾难） | 提出将离散词映射为连续实数向量（**分布式词嵌入 Word Embedding**），再用神经网络预测下一个词的条件概率分布 $P(w_t \mid w_{1:t-1})$ | `self.embed_tokens = nn.Embedding(vocab_size, hidden_size)` |
| **P02** | **Neural Machine Translation of Rare Words with Subword Units (BPE)**<br>Sennrich et al., *ACL 2016* ([arXiv:1508.07909](https://arxiv.org/abs/1508.07909)) | 词级（Word-level）词表遇到罕见词或派生词全部变成 `<unk>` | 将数据压缩中的 **Byte Pair Encoding** 引入 NLP：从基本字符出发，迭代合并语料中最高频的相邻符号对 $(a, b) \to ab$，自动学出高频词与词根子词 | [trainer/train_tokenizer.py](../minimind/trainer/train_tokenizer.py) |
| **P03** | **Language Models are Unsupervised Multitask Learners (GPT-2)**<br>Radford et al., *OpenAI 2019* | Unicode 字符集超过 14 万，字符级 BPE 基础词表依然过大且仍有 OOV | 提出 **Byte-Level BPE**：直接在 UTF-8 的 **256 个基础字节**上做 BPE 并配合正则预切分，使任意语言文本在仅数千/数万词表下实现 **0 未登录词（Zero OOV）**；确立 Decoder-Only 自回归预训练范式 | `pre_tokenizers.ByteLevel` + `tinymind/tokenizer_bpe.py` |

### 阶段二：现代 Transformer 架构、MoE 与解码（第 4–10 章）

| 序号 | 论文标题、作者与发表信息 | 解决的核心痛点 | 论文核心思想与公式 | MiniMind 对应实现 |
| :---: | :--- | :--- | :--- | :--- |
| **P04** | **Attention Is All You Need**<br>Vaswani et al., *NeurIPS 2017* ([arXiv:1706.03762](https://arxiv.org/abs/1706.03762)) | RNN/LSTM 必须按时间步串行计算，无法在 GPU 上并行且长距离依赖遗忘 | 提出完全基于 **缩放点积注意力** $\text{ Attention}(Q,K,V)=\text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$、多头机制（MHA）、残差连接与逐位置前馈网络的 Transformer 架构 | [model/model_minimind.py](../minimind/model/model_minimind.py) 整体骨架 |
| **P05** | **On Layer Normalization in the Transformer Architecture (Pre-LN)**<br>Xiong et al., *ICML 2020* ([arXiv:2002.04745](https://arxiv.org/abs/2002.04745)) | 原版 Post-LN Transformer 在深层网络中输出层附近梯度极大、浅层梯度消失，对 Warmup 极度敏感 | 理论证明将归一化移到残差分支内部（**Pre-LN**：$\mathbf{x}_{l+1} = \mathbf{x}_l + F_l(\text{LN}(\mathbf{x}_l))$）可使梯度范数在所有层保持同阶，残差主干形成恒等映射通路 | `MiniMindBlock.forward`（第 191–199 行） |
| **P06** | **Root Mean Square Layer Normalization (RMSNorm)**<br>Zhang & Sennrich, *NeurIPS 2019* ([arXiv:1910.07467](https://arxiv.org/abs/1910.07467)) | LayerNorm 需计算均值和方差两次规约，占用了可观的访存开销 | 证明 LayerNorm 的成功源于**缩放不变性（Rescaling Invariance）**而非平移不变性；去掉减均值和偏置 $\beta$，仅用均方根 $\text{RMS}(\mathbf{x})=\sqrt{\frac{1}{d}\sum x_i^2 + \epsilon}$ 归一化，提速 10%~30% | `class RMSNorm`（第 50–60 行） |
| **P07** | **GQA: Training Generalized Multi-Query Transformer Models**<br>Ainslie et al., *EMNLP 2023* ([arXiv:2305.13245](https://arxiv.org/abs/2305.13245)) | 自回归解码受限于加载 KV Cache 的显存带宽；MHA 显存太大，而 MQA（单 KV 头）质量下降明显 | 折中 MHA 与 MQA：将 $H_q$ 个 Query 头分成 $G$ 组，每组共享 1 对 Key/Value 头（**Grouped-Query Attention**），在几乎不损失精度的前提下将 KV Cache 压缩 $H_q/G$ 倍 | `Attention` + `repeat_kv`（8 个 Q 头共享 4 个 KV 头） |
| **P08** | **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**<br>Dao et al., *NeurIPS 2022* ([arXiv:2205.14135](https://arxiv.org/abs/2205.14135)) | 标准注意力在 HBM 显存上读写 $N \times N$ 中间矩阵，显存 $O(N^2)$ 且受限于 HBM 带宽 | 利用 GPU **SRAM 分块（Tiling）+ Online Softmax 递推重缩放**，不把 $N \times N$ 注意力矩阵写回 HBM，在无近似误差下实现 $O(N)$ 显存与数倍加速 | `F.scaled_dot_product_attention`（第 126 行） |
| **P09** | **RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE)**<br>Su et al., *Neurocomputing 2024 / 2021* ([arXiv:2104.09864](https://arxiv.org/abs/2104.09864)) | 绝对位置编码外推性差，而传统相对位置编码无法兼容线性注意力且破坏 KV 缓存形式 | 将 $q, k$ 的每两个维度视为复数，乘以位置旋转因子 $e^{im\theta_j}$，使得 $\langle R_m q, R_n k \rangle = \operatorname{Re}(q \bar{k} e^{i(m-n)\theta})$ **用绝对位置实现相对位置内积** | `precompute_freqs_cis` & `apply_rotary_pos_emb`（第 62–84 行） |
| **P10** | **YaRN: Efficient Context Window Extension of Large Language Models**<br>Peng et al., *ICLR 2024* ([arXiv:2309.00071](https://arxiv.org/abs/2309.00071)) | 均匀位置插值（PI）会挤压高频维度，导致模型丢失局部相邻 token 的先后分辨能力 | **NTK-by-parts 高低频分治**：波长远小于训练长度的高频维度不插值（$\gamma=0$），波长超限的低频维度线性插值（$\gamma=1$），中频用斜坡 `ramp` 平滑过渡 | `precompute_freqs_cis` 中的 `rope_scaling` 分支（第 64–73 行） |
| **P11** | **GLU Variants Improve Transformer (SwiGLU)**<br>Shazeer, *2020* ([arXiv:2002.05202](https://arxiv.org/abs/2002.05202)) | 传统两层 ReLU/GELU FFN 的特征表达效率有限 | 将门控线性单元（GLU）与 Swish/SiLU 激活结合：$\text{SwiGLU}(x) = (\text{SiLU}(xW_g) \odot xW_u)W_d$，在同等计算量下显著降低预训练困惑度 | `class FeedForward`（第 136–146 行） |
| **P12** | **MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases**<br>Liu et al., *ICML 2024* ([arXiv:2402.14905](https://arxiv.org/abs/2402.14905)) | 十亿参数以下的小模型该如何分配「深度 vs 宽度」以及是否共享输入输出词嵌入？ | 系统证明在小模型上 **Weight Tying（输入输出 Embedding 共享）** 可省下可观参数并提升精度，同时揭示了小模型深度与宽度的甜点折中法则 | `MiniMindForCausalLM` 的 `tie_word_embeddings` 与配置设计 |
| **P13** | **Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity**<br>Fedus et al., *JMLR 2022* ([arXiv:2101.03961](https://arxiv.org/abs/2101.03961)) | 传统 MoE 要求 Top-$k (k \ge 2)$ 才能求导，通信与计算开销大，且容易发生专家负载坍塌 | 提出 **Top-1 单专家路由**可保持质量并最小化开销，以及可微的**负载均衡辅助损失** $\mathcal{L}_{\text{aux}} = \alpha N \sum_{i=1}^N f_i P_i$ 防止赢家通吃 | `class MOEFeedForward`（第 148–181 行） |
| **P14** | **The Curious Case of Neural Text Degeneration (Nucleus / Top-p Sampling)**<br>Holtzman et al., *ICLR 2020* ([arXiv:1904.09751](https://arxiv.org/abs/1904.09751)) | 束搜索/贪心解码极易陷入无意义重复；固定 Top-K 在平坦分布时太窄、在尖锐分布时又把长尾怪词放进来 | 提出 **核采样（Top-p）**：动态截取累积概率达到阈值 $p$ 的最小词集合 $\sum_{x \in V^{(p)}} P(x) \ge p$，自动适配每一步分布的平坦或尖锐程度 | `MiniMindForCausalLM.generate`（第 279–282 行） |

### 阶段三 & 四：指令微调、LoRA 与知识蒸馏（第 11–15 章）

| 序号 | 论文标题、作者与发表信息 | 解决的核心痛点 | 论文核心思想与公式 | MiniMind 对应实现 |
| :---: | :--- | :--- | :--- | :--- |
| **P15** | **Training language models to follow instructions with human feedback (InstructGPT)**<br>Ouyang et al., *NeurIPS 2022* ([arXiv:2203.02155](https://arxiv.org/abs/2203.02155)) | 预训练基座模型只会续写文本（可能把用户的问题接着往下编问题），不理解「回答指令」的意图 | 确立现代后训练三步曲：**SFT（仅对示范回答计算交叉熵） $\to$ 偏好奖励建模 $\to$ RLHF 强化学习对齐** | `SFTDataset` + `train_full_sft.py` + `train_ppo.py` |
| **P16** | **LoRA: Low-Rank Adaptation of Large Language Models**<br>Hu et al., *ICLR 2022* ([arXiv:2106.09685](https://arxiv.org/abs/2106.09685)) | 全参数微调需要为每个下游任务保存一份完整模型权重，且优化器状态显存巨大 | 基于微调增量的低内在秩性质，冻结预训练矩阵 $W_0$，注入低秩旁路 $\Delta W = \frac{\alpha}{r}BA$（$A$ 高斯、$B$ 全零初始化），推理前可合并回 $W_0$ 实现零延迟 | [model/model_lora.py](../minimind/model/model_lora.py) + [trainer/train_lora.py](../minimind/trainer/train_lora.py) |
| **P17** | **Distilling the Knowledge in a Neural Network**<br>Hinton, Vinyals & Dean, *2015* ([arXiv:1503.02531](https://arxiv.org/abs/1503.02531)) | 独热（One-Hot）硬标签丢失了教师模型对各个候选词之间相似度的细粒度概率结构 | 引入带温度 $T$ 的软目标 $\mathbf{p}(T)=\text{softmax}(\mathbf{z}/T)$（**暗知识 Dark Knowledge**），并证明软标签 KL 散度需乘以 **$T^2$** 以补偿 $1/T^2$ 的梯度衰减 | [trainer/train_distillation.py](../minimind/trainer/train_distillation.py) |

### 阶段五 & 六：强化学习对齐（DPO / PPO / GRPO / CISPO）、Agent 与新架构（第 16–20 章）

| 序号 | 论文标题、作者与发表信息 | 解决的核心痛点 | 论文核心思想与公式 | MiniMind 对应实现 |
| :---: | :--- | :--- | :--- | :--- |
| **P18** | **Direct Preference Optimization: Your Language Model is Secretly a Reward Model (DPO)**<br>Rafailov et al., *NeurIPS 2023* ([arXiv:2305.18290](https://arxiv.org/abs/2305.18290)) | PPO 需要先训 Reward Model 再同时跑 4 个模型在线采样，显存开销大且超参极难调 | 利用带 KL 约束 RL 最优解的解析关系 $r(x,y)=\beta\log\frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)}+\beta\log Z(x)$ 代入 Bradley-Terry 模型消去配分函数 $Z(x)$，直接用二元交叉熵优化策略 | [trainer/train_dpo.py](../minimind/trainer/train_dpo.py) |
| **P19** | **Proximal Policy Optimization (PPO) & Generalized Advantage Estimation (GAE)**<br>Schulman et al., *ICLR 2016 / 2017* ([arXiv:1506.02438](https://arxiv.org/abs/1506.02438), [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)) | 传统策略梯度步长过大会导致策略一步崩溃，且单步 TD 与蒙特卡洛回报存在偏差-方差两难 | 用 **GAE** $\hat{A}_t = \sum_{l=0}^\infty (\gamma\lambda)^l \delta_{t+l}$ 平衡优势估计的偏差与方差；用 **Clipped Surrogate Objective** $\min(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon)A_t)$ 限制单步策略更新幅度 | [trainer/train_ppo.py](../minimind/trainer/train_ppo.py) |
| **P20** | **DeepSeekMath (GRPO) & DeepSeek-R1**<br>Shao et al., *2024* ([arXiv:2402.03300](https://arxiv.org/abs/2402.03300)); DeepSeek-AI, *2025* ([arXiv:2501.12948](https://arxiv.org/abs/2501.12948)) | PPO 中的 Critic 价值模型与 Actor 一样大，显存翻倍且在长思维链推理任务上难以准确拟合每个 token 的价值 | 提出 **GRPO**：对同一问题采样 $G$ 个回答，直接用**组内标准化奖励** $A_i = \frac{R_i - \mu_G}{\sigma_G}$ 代替 Critic 网络；DeepSeek-R1 进一步证明纯 RL + 可验证奖励可激发 `<think>` 自我反思能力 | [trainer/train_grpo.py](../minimind/trainer/train_grpo.py) |
| **P21** | **MiniMax-M1: Scaling Test-Time Compute Efficiently (CISPO)**<br>MiniMax, *2025* ([arXiv:2506.13585](https://huggingface.co/papers/2506.13585)) | 在 PPO/GRPO 的 token-level clip 中，低概率的关键反思转折词一旦 $r_t > 1+\epsilon$ 就会被直接抹掉梯度 | 提出 **CISPO**：不对梯度本身做零截断，而是对重要性采样比率做 stop-gradient 上限裁剪 $\mathcal{L} = -\text{sg}(\min(r_t, \epsilon_{\text{high}})) A_t \log \pi_\theta$，始终保留所有 token 的梯度贡献 | `train_grpo.py` & `train_agent.py` 中的 `loss_type="cispo"` |
| **P22** | **ReAct / Toolformer & LLaDA (Discrete Diffusion) / Linear Attention**<br>Yao et al. *ICLR 2023*; Schick et al. *NeurIPS 2023*; Nie et al. *2025* ([arXiv:2502.09992](https://arxiv.org/abs/2502.09992)) | 纯文本模型无法精确计算或获取实时信息；自回归逐词生成无法并行；二次方注意力难以扩展到超长序列 | **ReAct/Agentic RL**：将 `思考 -> 工具调用 -> 环境观察` 纳入多轮强化学习（对环境返回置 `mask=0`）；**dLM** 用双向掩码迭代去噪实现并行生成；**Linear Attention** 用核结合律将复杂度降为 $O(Ld^2)$ | [trainer/train_agent.py](../minimind/trainer/train_agent.py) + Discussions #618 / #704 |
