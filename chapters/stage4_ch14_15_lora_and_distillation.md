# 第四阶段（第 14–15 章）：参数高效微调（LoRA）与白盒知识蒸馏（KD）

> **本阶段目标**：
> 1. 不依赖 HuggingFace `peft` 库，只用 **70 行纯 PyTorch** 从零实现 LoRA 的定义、动态挂载、参数冻结、存取与无损权重合并（`model_lora.py` + `train_lora.py`）。
> 2. 彻底分清大模型里的「黑盒蒸馏（数据蒸馏）」与「白盒蒸馏（Logits KL 蒸馏）」，推导为什么软标签 KL 散度前面要乘以温度的平方 $T^2$（`train_distillation.py`）。
>
> **配套代码与练习**：
> - 参考实现：[tinymind/lora.py](../tinymind/lora.py)、[tinymind/distill.py](../tinymind/distill.py)
> - 动手练习：[labs/lora.py](../labs/lora.py)（`TODO(ch14-1)` ~ `TODO(ch14-3)`）、[labs/distill.py](../labs/distill.py)（`TODO(ch15-1)`）
> - 对照 MiniMind 源码：[minimind/model/model_lora.py](../minimind/model/model_lora.py)、[minimind/trainer/train_lora.py](../minimind/trainer/train_lora.py)、[minimind/trainer/train_distillation.py](../minimind/trainer/train_distillation.py)
> - 闯关评测：`python tools/grade.py ch14`、`python tools/grade.py ch15`

---

## 第 14 章：70 行从零手写 LoRA（Low-Rank Adaptation）

### 14.1 为什么微调不需要更新整个大矩阵？（内在秩假说）

Aghajanyan et al. (2020) 和 Hu et al. (2021, *LoRA*) 发现：当一个预训练好的大模型适应某个下游任务（比如医疗问答、自我认知）时，权重矩阵的变化量 $\Delta W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ 具有极低的「内在秩（Intrinsic Rank）」。
因此，我们**冻结原始权重 $W_0$**，在旁边并联一条由两个瘦长矩阵 $A \in \mathbb{R}^{r \times d_{\text{in}}}$（先降维到 $r$）和 $B \in \mathbb{R}^{d_{\text{out}} \times r}$组成的低秩旁路（其中秩 $r \ll \min(d_{\text{in}}, d_{\text{out}})$，比如 $d=768, r=16$）：

$$\mathbf{h} = W_0 \mathbf{x} + \Delta W \mathbf{x} = W_0 \mathbf{x} + \frac{\alpha}{r} B A \mathbf{x}$$

- **参数量省了多少？**
  原本一个 $768 \times 768$ 的方阵有 $589,824$ 个参数；换成 $r=16$ 的 $A, B$ 后只有 $16 \times (768 + 768) = 24,576$ 个参数，**仅为原来的 $4.17\%$**！而且因为主干参数不求梯度，优化器（AdamW 的一阶/二阶动量）显存也省掉了 $95\%$ 以上。

### 14.2 三个必须掌握的数学与代码细节（对照 [model/model_lora.py](../minimind/model/model_lora.py)）

#### 细节 1：为什么 $A$ 用高斯初始化，而 $B$ 必须全 0 初始化？（第 15–17 行）
```python
self.A.weight.data.normal_(mean=0.0, std=0.02)
self.B.weight.data.zero_()
```
- 如果 $A$ 和 $B$ 都随机初始化，在刚挂上 LoRA 的第 0 步，$\Delta W = B A \ne 0$，相当于往已经训好的 SFT 模型里突然注入了一堆随机噪声，前几步就会破坏原有的特征表示。
- 反之，如果 $A$ 和 $B$ **都**初始化为 0，那么根据链式法则，$\frac{\partial \mathcal{L}}{\partial A} = B^T \frac{\partial \mathcal{L}}{\partial \mathbf{y}} = 0$，$\frac{\partial \mathcal{L}}{\partial B} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} (A\mathbf{x})^T = 0$，两边梯度全为 0，**永远处于鞍点训不动**！
- **一面高斯 ($A$)、一面全零 ($B$)**：既保证了第 0 步前向输出 $\Delta W \mathbf{x} = B(A\mathbf{x}) = \mathbf{0}$（**训练起点与原模型 100% 等价**），又因为 $A\mathbf{x} \ne \mathbf{0}$，第 1 步反向传播时 $B$ 就能立刻收到非零梯度 $\frac{\partial \mathcal{L}}{\partial B} \ne 0$，第 2 步起 $B \ne 0$，$A$ 也开始更新！

#### 细节 2：Python 闭包晚绑定陷阱与默认参数显式绑定（第 23–34 行）
看看 MiniMind 是怎么在不修改 `model_minimind.py` 一行代码的情况下，动态给模型挂上 LoRA 的：
```python
def apply_lora(model, rank=16, lora_alpha=None):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.in_features == module.out_features:
            lora = LoRA(module.in_features, module.out_features, rank=rank, lora_alpha=lora_alpha).to(model.device)
            setattr(module, "lora", lora)
            original_forward = module.forward

            # 显式绑定
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)

            module.forward = forward_with_lora
```
- **为什么选择 `module.in_features == module.out_features`？**
  在 MiniMind 默认配置（$H=768, H_q=8, H_{kv}=4, d_h=96$）中，所有线性层里满足输入维度等于输出维度的方阵正好是每层 Attention 的 **`q_proj` ($768 \to 768$) 和 `o_proj` ($768 \to 768$)**！
- **为什么 `def forward_with_lora(x, layer1=original_forward, layer2=lora):` 要把两个函数写成默认参数？**
  这是 Python 面试与工程中最经典的**循环闭包晚绑定（Late Binding）陷阱**！如果写成 `def forward_with_lora(x): return original_forward(x) + lora(x)`，由于 Python 闭包在函数**被调用时**才去外层作用域查找变量 `original_forward` 和 `lora`，循环结束后所有层的 `forward_with_lora` 都会指向**循环最后一层**的那个 `original_forward` 和 `lora`！而默认参数是在 `def` **定义的那一刻求值绑定**的，从而把当前层的 `original_forward` 和 `lora` 牢牢锁死在该函数对象上。

#### 细节 3：零推理延迟的权重合并 `merge_lora`（第 59–70 行）
由于矩阵乘法满足分配律：
$$W_0 \mathbf{x} + \frac{\alpha}{r} B A \mathbf{x} = \underbrace{\left(W_0 + \frac{\alpha}{r} B A\right)}_{W_{\text{merged}}} \mathbf{x}$$
训练结束后，直接算出增量矩阵 `delta = (B.weight @ A.weight) * scaling` 加回 `module.weight` 并保存为普通 `.pth` 文件，推理时就完全不需要 `LoRA` 旁路，**推理速度与原模型一模一样（零额外开销）**！

---

## 第 15 章：知识蒸馏 —— 黑盒蒸馏 vs 白盒蒸馏

### 15.1 两种蒸馏范式

当你想用百亿/千亿的大模型（Teacher，或 MiniMind 的 16 层/MoE 大模型）来提升 64M 小模型（Student）的能力时，有两条路线：

1. **黑盒蒸馏（数据蒸馏 / SFT 蒸馏）**：
   只拿 Teacher 生成的文本回答 $y_{\text{teacher}}$（包括 `<think>...</think>` 思维链或 `<tool_call>` 调用轨迹）放进 `sft_t2t.jsonl`，让 Student 用普通交叉熵损失做 SFT：
   $$\mathcal{L}_{\text{blackbox}} = \text{CE}(y_{\text{teacher}},\, p_{\text{student}})$$
   MiniMind 主线 SFT 数据中混入的 `Qwen3-4B` 和 `DeepSeek-R1` 合成数据就属于黑盒蒸馏。
2. **白盒蒸馏（Logits 分布蒸馏，Hinton et al. 2015）**：
   在每一步预测下一个 token 时，不仅让 Student 看静态的 one-hot 标准答案，还把 Teacher 输出层在整个词表（6400 个词）上的**软概率分布（Soft Targets）**拿过来，用 KL 散度逼近它！

### 15.2 白盒蒸馏数学公式与 $T^2$ 缩放因子推导

查看 [minimind/trainer/train_distillation.py#L24-L39, L74-L98](../minimind/trainer/train_distillation.py#L24-L39)：
```python
def distillation_loss(student_logits, teacher_logits, temperature=1.0, reduction='batchmean'):
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    kl = F.kl_div(student_log_probs, teacher_probs, reduction=reduction)
    return (temperature ** 2) * kl
```
总损失是硬标签交叉熵 $\mathcal{L}_{\text{CE}}$ 与软标签蒸馏损失 $\mathcal{L}_{\text{KD}}$ 的加权和：

$$\mathcal{L}_{\text{total}} = \alpha \cdot \mathcal{L}_{\text{CE}}(\mathbf{y}, \mathbf{z}^{(s)}) + (1 - \alpha) \cdot T^2 \cdot \mathrm{KL}\left(\text{softmax}\left(\frac{\mathbf{z}^{(t)}}{T}\right) \;\Bigg\|\; \text{softmax}\left(\frac{\mathbf{z}^{(s)}}{T}\right)\right)$$

**两个核心原理**：
1. **为什么要用温度 $T > 1$（如 $T = 1.5$ 或 $2.0$）软化分布？**
   当 $T=1$ 时，一个训练好的 Teacher 模型对第一名候选词的概率可能是 $0.999$，对近义词是 $0.0009$，对无关词是 $10^{-9}$——近义词和无关词之间的珍贵相对关系（Hinton 称之为 **暗知识 Dark Knowledge**）在数值上太小，对梯度几乎没有贡献。除以 $T > 1$ 后，分布被抹平，近义词的概率被放大到比如 $0.15$，Student 就能清晰地学到「在这个语境下，除了标准答案，还有哪些词也是通顺的」。
2. **为什么最后要乘以 `(temperature ** 2)`（$T^2$）？**
   对软化后的 KL 散度关于学生 logit $z^{(s)}_i$ 求偏导：
   $$\frac{\partial}{\partial z^{(s)}_i} \mathrm{KL}\left(\mathbf{p}^{(t)}(T) \parallel \mathbf{p}^{(s)}(T)\right) = \frac{1}{T} \left(p^{(s)}_i(T) - p^{(t)}_i(T)\right)$$
   当 $T$ 较大时，利用一阶泰勒展开 $e^{z/T} \approx 1 + z/T$，可以发现括号里的 $(p^{(s)}_i(T) - p^{(t)}_i(T))$ 本身又与 $\frac{1}{T}$ 成正比！两者相乘导致 **KL 项的梯度大小会以 $\frac{1}{T^2}$ 的速度缩小**！
   为了在调节超参数 $T$ 时不让蒸馏项的梯度相对于 $\mathcal{L}_{\text{CE}}$ 突然缩小 $T^2$ 倍，必须乘上 $T^2$ 把梯度量级补偿回来。

---

## 🛠️ 本阶段动手实验（Labs）

```bash
# 1. 完成 labs/lora.py 中的 TODO(ch14-1) ~ TODO(ch14-3)
python tools/grade.py ch14 -v

# 2. 完成 labs/distill.py 中的 TODO(ch15-1)
python tools/grade.py ch15 -v
```
