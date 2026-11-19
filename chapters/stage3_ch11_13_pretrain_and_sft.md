# 第三阶段（第 11–13 章）：从零预训练（Pretrain）到全参数指令微调（Full SFT）

> **本阶段目标**：掌握训练一个「能对话的 MiniMind Zero 模型」仅有的两个必选阶段：**Pretrain（学会续写与知识）** 与 **Full SFT（学会按角色对话、思考与调用工具）**。你会看清这两者在模型和训练循环上几乎一模一样，唯一的本质区别在于 **Loss Mask（哪些 token 参与计算损失）**！同时掌握现代 PyTorch 大模型训练的完整工程套件（余弦退火、混合精度、梯度累积、梯度裁剪、原子检查点与跨卡断点续训）。
>
> **配套代码与练习**：
> - 参考实现：[tinymind/data.py](../tinymind/data.py)、[tinymind/train_utils.py](../tinymind/train_utils.py)
> - 动手练习：[labs/data.py](../labs/data.py)（`TODO(ch11-1)`, `TODO(ch13-1)`）、[labs/train_utils.py](../labs/train_utils.py)（`TODO(ch12-1)` ~ `TODO(ch12-3)`）
> - 对照 MiniMind 源码：[minimind/dataset/lm_dataset.py](../minimind/dataset/lm_dataset.py)（第 1–123 行）、[minimind/trainer/train_pretrain.py](../minimind/trainer/train_pretrain.py)、[minimind/trainer/train_full_sft.py](../minimind/trainer/train_full_sft.py)、[minimind/trainer/trainer_utils.py](../minimind/trainer/trainer_utils.py)
> - 闯关评测：`python tools/grade.py ch11`、`python tools/grade.py ch12`

---

## 第 11 章：预训练数据集与数据工程（`PretrainDataset`）

### 11.1 `PretrainDataset` 的 8 行核心实现

打开 [minimind/dataset/lm_dataset.py#L40-L58](../minimind/dataset/lm_dataset.py#L40-L58)：
```python
class PretrainDataset(Dataset):
    ...
    def __getitem__(self, index):
        sample = self.samples[index]
        tokens = self.tokenizer(str(sample['text']), add_special_tokens=False,
                                max_length=self.max_length - 2, truncation=True).input_ids
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels
```
让我们用一个长度为 `max_length = 8` 的例子看清 `input_ids` 和 `labels` 的每一个位置：
假设文本分词后为 `[105, 230, 88]`，且 `bos=1, eos=2, pad=0`：
```text
位置 idx :    0      1      2      3      4      5      6      7
input_ids: [  1,   105,   230,    88,     2,     0,     0,     0 ]
labels   : [ -100, 105,   230,    88,     2,  -100,  -100,  -100 ]  (注: pad=0 的位置全变 -100)
```
进入模型 `forward` 后，做错位切片 `x = logits[:, :-1]`，`y = labels[:, 1:]`：
- 位置 0 (`bos=1`) 预测 `y[0] = 105`（句首第一个词）
- 位置 1 (`105`) 预测 `y[1] = 230`
- 位置 2 (`230`) 预测 `y[2] = 88`
- 位置 3 (`88`) 预测 `y[3] = 2 (eos)`（**学会在句子结束时输出终止符！**）
- 位置 4..6 的目标 `y[4..6]` 全部是 `-100`，被 `F.cross_entropy(..., ignore_index=-100)` 自动跳过！

### 11.2 拓展知识：预训练语料是如何清洗出来的？

MiniMind 的 `requirements.txt` 中出现了 `datasketch` 和 `simhash`。在大规模预训练数据工程中，原始网页/语料必须经过以下 4 道流水线才能变成高质量的 `pretrain_t2t.jsonl`：
1. **启发式过滤（Heuristic Filtering）**：过滤过短文本、标点/数字占比异常的文本、含大段重复 n-gram 的垃圾文本。
2. **模糊去重（Fuzzy Deduplication）**：利用 **MinHash LSH（`datasketch`）** 或 **SimHash（汉明距离）** 剔除高度相似的文档。去重对小模型尤其致命——如果同一段话在语料里重复 100 次，64M 的小模型会直接把容量浪费在死记硬背这段话上，并导致推理时严重复读。
3. **合成与知识蒸馏语料配比**：混入教科书级合成文本（如 Cosmopedia / Magpie / DeepCtrl），提高知识密度。
4. **`max_seq_len` 权衡**：MiniMind 在中文上约 $1.5 \sim 1.7$ 字符/token。`max_seq_len` 设太小会截断长句语义，设太大而样本偏短则大部分算力浪费在 `pad_token` 的注意力矩阵上（进一步进阶可以把多条短文档用 `<|im_end|><|im_start|>` 拼接成满长度的 **Sequence Packing**）。

---

## 第 12 章：现代 LLM 训练循环工程（`train_pretrain.py` & `trainer_utils.py`）

打开 [minimind/trainer/train_pretrain.py](../minimind/trainer/train_pretrain.py)，你会看到一个极其干净、只有 163 行的工业级 PyTorch 训练脚本。它包含 **6 个核心工程组件**：

### 12.1 余弦退火学习率调度（`get_lr`）
查看 [trainer_utils.py#L42-L43](../minimind/trainer/trainer_utils.py#L42-L43)：
```python
def get_lr(current_step, total_steps, lr):
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))
```
$$\eta(t) = \eta_0 \cdot \left(0.1 + 0.45 \left(1 + \cos\frac{\pi t}{T}\right)\right)$$
- 当 $t = 0$ 时，$\cos 0 = 1$，$\eta(0) = \eta_0 \cdot (0.1 + 0.9) = \eta_0$（预训练默认 $5\times 10^{-4}$）；
- 当 $t = T$ 时，$\cos \pi = -1$，$\eta(T) = 0.1 \eta_0$（平滑衰减到初始学习率的 $10\%$）。
注意在 [train_pretrain.py#L29-L31](../minimind/trainer/train_pretrain.py#L29-L31) 中，学习率是**每个 step（每个 batch）动态更新**的，而不是每个 epoch 更新一次。

### 12.2 混合精度训练：`bfloat16` vs `float16` + `GradScaler`
查看 [train_pretrain.py#L112-L114, L129](../minimind/trainer/train_pretrain.py#L112-L114)：
- **`float16`**：1 位符号 + **5 位指数** + 10 位尾数。最大正数只有 $65504$，极小的梯度（$< 6\times 10^{-8}$）会直接下溢成 0！因此用 `float16` 时**必须开启 `GradScaler`**：反向传播前先把 `loss` 乘以一个大倍数（如 $2^{16}$）把梯度抬高，更新参数前再 `scaler.unscale_(optimizer)` 除回来。
- **`bfloat16`（Brain Float 16，MiniMind 默认）**：1 位符号 + **8 位指数（与 `float32` 完全相同的动态范围！）** + 7 位尾数。不会像 fp16 那样动不动溢出，因此 `enabled=(args.dtype == 'float16')` 在 bf16 下会自动把 `GradScaler` 变成无操作的直通模式。

### 12.3 梯度累积（Gradient Accumulation）与梯度裁剪（Grad Clip）
查看 [train_pretrain.py#L33-L47](../minimind/trainer/train_pretrain.py#L33-L47)：
```python
with autocast_ctx:
    res = model(input_ids, labels=labels)
    loss = res.loss + res.aux_loss
    loss = loss / args.accumulation_steps

scaler.scale(loss).backward()

if step % args.accumulation_steps == 0 or step == iters:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
```
1. **为什么要写 `loss = loss / args.accumulation_steps`？**
   当单卡显存只能放下 `batch_size = 32`，但我们想要 `256` 的等效大 Batch 时，可以设 `accumulation_steps = 8`：连续做 8 次 `backward()`（PyTorch 默认会把梯度累加在 `p.grad` 上），第 8 步才调一次 `optimizer.step()` 和 `zero_grad()`。因为单个小 batch 的 `res.loss` 是小 batch 内的均值，8 次累加后梯度会放大 8 倍，所以每次反向传播前要先除以 `accumulation_steps`（打印日志时再乘回来，见第 51 行）。
2. **为什么要调 `clip_grad_norm_(..., 1.0)`？**
   把所有参数拼接成的全局梯度向量的 $L_2$ 范数限制在 `1.0` 以内：若 $\|\mathbf{g}\|_2 > 1$，则 $\mathbf{g} \leftarrow \mathbf{g} / \|\mathbf{g}\|_2$。一旦遇到某条异常脏数据产生巨幅梯度，不会一脚把训练好的权重踹飞。
3. **`optimizer.zero_grad(set_to_none=True)`**：把 `p.grad` 直接置为 `None` 而不是填 0 张量，既省显存又少一次写内存操作。

### 12.4 原子检查点与跨 GPU 数量无缝续训
查看 [trainer_utils.py#L65-L117, L135-L159](../minimind/trainer/trainer_utils.py#L65-L117)：
- **先写 `.tmp` 再 `os.replace(ckp_tmp, ckp_path)`**：如果直接往目标路径 `torch.save`，写到一半服务器断电或 Ctrl+C，原有的好权重就彻底损坏了！POSIX 系统下 `os.replace` 是原子重命名操作，保证磁盘上的权重文件永远是完整的。
- **保存两份文件**：
  1. `out/pretrain_768.pth`：只存 `fp16` 的纯净模型 `state_dict`（约 128MB），体积小，直接用于推理或下一阶段 SFT 加载。
  2. `checkpoints/pretrain_768_resume.pth`：存 `fp32` 模型权重 + `optimizer` 的一阶/二阶动量状态 + `scaler` + `epoch/step/world_size/wandb_id`。
- **跨 GPU 数量恢复 + `SkipBatchSampler`**：如果上次用 4 张卡训到 `step=1000`（每张卡跑了 1000 步，相当于全局跑了 4000 个单卡 batch），这次改用 2 张卡续训，第 114 行会自动把 `step` 换算为 $1000 \times 4 / 2 = 2000$，并用 `SkipBatchSampler` 在固定随机种子下秒级跳过前 2000 个 batch，接着往下训！

---

## 第 13 章：全参数指令微调（Full SFT）与 Loss Mask 机制

### 13.1 Pretrain 与 SFT 的唯一核心区别：只对 Assistant 回复算 Loss！

对比 [train_pretrain.py](../minimind/trainer/train_pretrain.py) 和 [train_full_sft.py](../minimind/trainer/train_full_sft.py)，你会惊讶地发现：**两个训练脚本的代码 98% 完全一样！**（唯一的超参变化是 SFT 学习率从 `5e-4` 降到了 `1e-6`，`max_seq_len` 从 `340` 升到了 `768`，并且初始权重加载 `from_weight='pretrain'`）。

那么，是什么让一个只会「无脑往下接龙」的基座模型变成了「一问一答的对话助手」？全部秘密都在 [dataset/lm_dataset.py#L61-L122](../minimind/dataset/lm_dataset.py#L61-L122) 的 **`SFTDataset.generate_labels`** 里！

```python
self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

def generate_labels(self, input_ids):
    labels = [-100] * len(input_ids)
    i = 0
    while i < len(input_ids):
        if input_ids[i:i + len(self.bos_id)] == self.bos_id:
            start = i + len(self.bos_id)
            end = start
            while end < len(input_ids):
                if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                    break
                end += 1
            for j in range(start, min(end + len(self.eos_id), self.max_length)):
                labels[j] = input_ids[j]
            i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
        else:
            i += 1
    return labels
```

#### 图解多轮对话中的 SFT Label 掩码
一段两轮对话渲染并分词后，`input_ids` 和 `generate_labels` 生成的 `labels` 对齐如下：

```text
Token 内容        : <|im_start|>user\n  你好  <|im_end|>\n  <|im_start|>assistant\n  我是助手  <|im_end|>\n  <|pad|>
input_ids        : [     1,     ...,   xxx,    2,  \n,        1,   assistant, \n,   我是助手,    2,  \n,       0   ]
labels (SFT掩码) : [  -100,    -100,  -100, -100, -100,    -100,      -100, -100,   我是助手,    2,  \n,    -100   ]
                                                                                    ▲──────────────────▲
                                                                          只有 [start, end + len(eos_id)) 保留！
```
**三个必须想通的问题**：
1. **为什么 `labels` 一开始全初始化成 `-100`？**
   这样 `system` 提示词、`user` 的提问、`<tool_response>` 环境返回结果，以及结尾的 `pad` 都默认不计入损失。模型只需要把有限的容量用在学习「当看到 `<|im_start|>assistant\n` 时，我该如何回答」。
2. **为什么匹配了多轮循环 `while i < len(input_ids)`？**
   在一段多轮对话（User 1 $\to$ Assistant 1 $\to$ User 2 $\to$ Assistant 2）中，这样可以在**一次前向传播**里同时对第 1 轮和第 2 轮的所有 Assistant 回复计算损失，训练效率比拆成多条单轮样本高得多！
3. **为什么循环上限是 `end + len(self.eos_id)`（把 `<|im_end|>\n` 也包含进 labels）？**
   如果你只训练到回答的最后一个汉字，而不把 `<|im_end|>` 放进 `labels`，模型就永远学不会在回答完毕时输出 `<|im_end|>`，推理时就会一直喋喋不休直到撞上 `max_new_tokens`！

### 13.2 数据增强：`pre_processing_chat` 与 `post_processing_chat`

查看 [dataset/lm_dataset.py#L9-L39](../minimind/dataset/lm_dataset.py#L9-L39)：
1. **`pre_processing_chat`（20% 概率随机挂载 System Prompt）**：
   大多数开源对话数据集第一条消息直接就是 `user`，没有 `system`。如果在训练时从来没见过 `system`，推理时用户一旦传了 System Prompt 模型就会懵；反之如果每条都固定加同一个 `system`，模型又会过拟合那个特定句子。所以 MiniMind 以 `20%` 概率从 10 条中英文通用 System Prompt 中随机抽一条插在最前面。
2. **`post_processing_chat`（80% 概率移除空 `<think>\n\n</think>\n\n`）**：
   我们在第 3 章看到，`chat_template` 会给没有 `reasoning_content` 的普通回复也默认加上空的 `<think>\n\n</think>\n\n`。在 SFT 时：
   - 对于**带真实思考过程**的数据：`<think>...</think>` 里面有内容，原样保留，模型学会显式推理；
   - 对于**不带思考过程**的数据：保留 **20%** 的空 `<think>\n\n</think>\n\n`（这样当推理时 `open_thinking=False` 模板自动注入空 `<think>\n\n</think>\n\n` 时，模型认识这种格式，能紧接着输出正文），剩下 **80%** 直接删掉空标签。

---

## 🛠️ 本阶段动手实验（Labs）

```bash
# 1. 完成 labs/data.py 中的 TODO(ch11-1) 与 TODO(ch13-1)
python tools/grade.py ch11 -v

# 2. 完成 labs/train_utils.py 中的 TODO(ch12-1) ~ TODO(ch12-3)
python tools/grade.py ch12 -v
```
完成后，你还可以直接运行 `python tools/train_tiny_pipeline.py`，亲眼观察阶段 1（Pretrain）和阶段 2（Full SFT）是如何在几秒钟内让一个随机初始化的模型学会回答 `"你是谁？" -> "我是MiniMind小助手。"` 的！
