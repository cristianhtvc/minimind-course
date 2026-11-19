# 第六阶段（第 19–20 章）：Tool Calling、Agentic RL 与全生态部署 & 前沿拓展

> **本阶段目标**：
> 1. 掌握大模型是如何从「纯文本聊天」进化为「能调用外部工具的智能体（Agent）」的：深入 `train_agent.py` 的**多轮交互 Rollout 状态机**、**多轮 `response_mask` 精确掩码**、**AST 抽象语法树安全算术求值**，以及**基于可验证 Ground Truth 的奖励函数（RLVR）**（第 19 章）。
> 2. 掌握 MiniMind 模型从训练完成到生产落地的全套工具链：权重转换为 HuggingFace `Llama/Qwen3` 格式、OpenAI 兼容 API 服务端、C-Eval/CMMLU 客观评测、vLLM / SGLang / llama.cpp (GGUF) / Ollama / MNN 端侧部署，以及 MiniMind 社区的两大前沿架构拓展——**离散扩散语言模型（dLM）** 与 **线性注意力（Linear Attention）**（第 20 章）。
>
> **配套代码与练习**：
> - 参考实现：[tinymind/agent.py](../tinymind/agent.py)
> - 动手练习：[labs/agent.py](../labs/agent.py)（`TODO(ch19-1)` ~ `TODO(ch19-2)`）
> - 对照 MiniMind 源码：[minimind/trainer/train_agent.py](../minimind/trainer/train_agent.py)、[minimind/trainer/trainer_utils.py#L182-L208](../minimind/trainer/trainer_utils.py#L182-L208)、[minimind/scripts/eval_toolcall.py](../minimind/scripts/eval_toolcall.py)、[minimind/scripts/convert_model.py](../minimind/scripts/convert_model.py)、[minimind/scripts/serve_openai_api.py](../minimind/scripts/serve_openai_api.py)
> - 闯关评测：`python tools/grade.py ch19`

---

## 第 19 章：Tool Calling 与 Agentic RL（`train_agent.py`）

### 19.1 大模型是怎么「调用函数」的？

大模型本身并不会执行 Python 代码或发 HTTP 请求。所谓 **Tool Calling（Function Calling）**，本质上是一套**协议约定 + 外部循环**：

1. **System 注入工具说明书**：`chat_template` 把 JSON Schema 格式的 `tools` 列表渲染进 `<|im_start|>system` 块中，包裹在 `<tools>...</tools>` 标签内，并告诉模型：「如需调用工具，请输出 `<tool_call>{"name": ..., "arguments": ...}</tool_call>`」。
2. **模型生成 `<tool_call>`**：例如用户问 `"帮我算一下 256 乘以 37"`，模型输出：
   ```xml
   <think>
   我应该调用 calculate_math 工具计算 256 * 37。
   </think>

   <tool_call>
   {"name": "calculate_math", "arguments": {"expression": "256 * 37"}}
   </tool_call><|im_end|>
   ```
3. **外部宿主拦截并执行**：宿主程序（如 `eval_toolcall.py` 或 `train_agent.py` 的 `rollout_single`）用正则 `r'<tool_call>(.*?)</tool_call>'` 提取出 JSON，在本地执行 `execute_tool("calculate_math", {"expression": "256 * 37"})`，拿到结果 `{"result": "9472"}`。
4. **回填 `<tool_response>` 进入下一轮**：宿主把结果作为一条 `role="tool"` 的消息追加到对话历史中（模板会将其渲染为 `<|im_start|>user\n<tool_response>\n{"result": "9472"}\n</tool_response><|im_end|>\n`），再次调用模型生成最终的自然语言回答 `"256 乘以 37 等于 9472。"`！

### 19.2 多轮 Agent Rollout 的核心难点：`response_mask` 怎么打？

在单轮 GRPO（第 18 章）中，序列只有两段：`[Prompt] + [Completion]`，只要对 `[Completion]` 算策略梯度就行了。
但在多轮 Agentic RL（[minimind/trainer/train_agent.py#L93-L159](../minimind/trainer/train_agent.py#L93-L159) 的 `rollout_single`）中，一条完整轨迹是**模型生成与环境返回交替拼接**而成的：

```text
[初始 Prompt] ──▶ [第1轮 Assistant 生成 <tool_call>] ──▶ [环境注入 <tool_response> + 第2轮引导头] ──▶ [第2轮 Assistant 生成最终回答]
   (不训练)                 mask = 1 (要训练!)                     mask = 0 (绝不能训练!)                  mask = 1 (要训练!)
```

打开 [train_agent.py#L106-L158](../minimind/trainer/train_agent.py#L106-L158)，看看 MiniMind 是怎么维护 `all_resp_ids`、`all_masks` 和 `all_old_logps` 的：
1. **模型自己采样出的第 $k$ 轮 token (`comp_ids`)**：
   - 追加进 `all_resp_ids`
   - 对应的 `all_masks` 追加 **全 `1`**（表示这些 token 是策略 $\pi_\theta$ 的动作，要计算 CISPO/GRPO 梯度）
   - 对应的 `all_old_logps` 追加采样时真实的 `per_token_logps`
2. **环境执行完工具后拼接到对话里的 `<tool_response>...</tool_response><|im_end|>\n<|im_start|>assistant\n<think>...`（即 `env_ids`，第 148–153 行）**：
   ```python
   if len(env_ids) > 0:
       all_resp_ids.extend(env_ids)
       all_masks.extend([0] * len(env_ids))
       all_old_logps.extend([0.0] * len(env_ids))
   ```
   - 对应的 `all_masks` 追加 **全 `0`**！因为环境返回的文本不是模型生成的，如果对它算策略梯度或 KL 散度，就会强迫模型去背诵工具返回的字符串，导致策略崩溃。

### 19.3 安全工程：`safe_math_eval` 为什么绝不用 `eval()`？

当模型在 RL 探索中随机生成 `calculate_math` 的参数 `expression` 时，如果直接调 Python 内置的 `eval(expression)`，会有两个致命风险：
1. **任意代码执行风险**：如果语料或提示里含有 `__import__('os').system(...)`，`eval` 会直接在你的服务器上执行命令。
2. **算力炸弹（DoS）**：如果模型随机吐出 `"9**9**9"` 或 `"10**9999999"`，一个大整数幂运算就会把 Python 进程卡死几个小时，整场 RL 训练直接挂起！

查看 [minimind/trainer/trainer_utils.py#L182-L208](../minimind/trainer/trainer_utils.py#L182-L208) 的 `safe_math_eval`：
- 它用 `ast.parse(expr, mode='eval')` 把表达式解析成抽象语法树（AST），只放行数字常量 `ast.Constant`、四则与幂运算 `ast.BinOp`、正负号 `ast.UnaryOp` 以及 `math` 库白名单函数；
- 特别设计了 `pow_guard(base, exp)`：在计算 $a^b$ 前先检查 $|b| > 10^4$ 或 $|b| \log_{10}|a| > 100$（即结果超过 $10^{100}$），一旦超限立刻抛出 `ValueError` 被工具捕获为错误返回，保护训练进程永不卡死！

### 19.4 可验证奖励（RLVR）设计（[train_agent.py#L185-L242](../minimind/trainer/train_agent.py#L185-L242)）

在 `agent_rl.jsonl` 中，每条数据带有一个标准答案列表 `gt`（Ground Truth，例如 `["9472"]`）。`calculate_rewards` 综合了三个维度的自动化信号：
1. **格式合规分**：检查每轮 `<tool_call>` 与 `</tool_call>` 是否成对闭合，每少闭合一个扣 `0.5` 分；
2. **工具调用对齐分**：检查调用的工具名是否在 `tools` 列表内、必填参数是否齐全（`CHECK_ARGS`），合法调用次数与 `len(gt)` 完全一致加 `+0.5` 分，否则按差值扣分；
3. **最终结果校验分（`validate_gt_in_text`）**：提取最终回答里的数字（自动忽略千分位逗号如 `9,472` 并按浮点数误差 $<10^{-6}$ 比对）或字符串，按命中 `gt` 的比例给予最高 **`+2.5` 分**！

---

## 第 20 章：模型转换、评测、API 服务与前沿拓展

当你走完前面的训练，手里有了 `out/full_sft_768.pth` 或 `out/grpo_768.pth`，如何把它接入主流大模型生态？

### 20.1 模型格式转换（`scripts/convert_model.py`）

MiniMind 默认保存的是纯 PyTorch `state_dict`（`.pth`），而 vLLM、Ollama、llama.cpp、LM-Evaluation-Harness 等第三方工具认识的是 **HuggingFace Transformers 目录格式（`config.json` + `model.safetensors` / `pytorch_model.bin` + `tokenizer.json`）**。
打开 [minimind/scripts/convert_model.py](../minimind/scripts/convert_model.py)，它提供了两个方向的转换函数：
1. **`convert_torch2transformers_minimind`**：保持 `MiniMindForCausalLM` 类，直接用 `register_for_auto_class("AutoModelForCausalLM")` 导出为支持 `trust_remote_code=True` 的 HuggingFace 文件夹。
2. **`convert_torch2transformers_llama` / Qwen3 兼容导出**：由于 MiniMind-3 的架构（RMSNorm + GQA + QK-Norm + RoPE + SwiGLU）与 **Qwen3 / Llama** 完全同构，并且层命名（`model.layers.i.self_attn.q_proj`、`mlp.gate_proj` 等）已经在 `model_minimind.py` 中 1:1 对齐，因此转换脚本可以直接将权重装进 HuggingFace 原生的 `LlamaConfig / Qwen3` 结构导出——**无需修改第三方推理框架一行代码，就能直接被 `vllm`、`sglang`、`llama.cpp` 原生加载！**

### 20.2 OpenAI 兼容 API 服务与 WebUI

- **[scripts/serve_openai_api.py](../minimind/scripts/serve_openai_api.py)**：用 FastAPI 实现了标准的 `POST /v1/chat/completions` 接口，支持：
  - `stream=True`（通过 `TextIteratorStreamer` + Server-Sent Events 实时推送 token）；
  - 自动将输出中的 `<think>...</think>` 拆分为 `reasoning_content` 字段，将 `<tool_call>...</tool_call>` 拆分为 OpenAI 标准 `tool_calls` 数组；
  - 可直接作为自定义模型接入 **Open-WebUI、FastGPT、Dify、Cursor / Continue** 或任何使用 `openai` Python/JS SDK 的应用。
- **[scripts/web_demo.py](../minimind/scripts/web_demo.py)**：基于 Streamlit 的可视化聊天界面，支持折叠展示思考过程与多轮工具调用可视化。

### 20.3 客观评测与端侧量化部署

1. **客观基准评测（`lm-evaluation-harness`）**：
   转换为 Transformers 格式后，可直接用 EleutherAI 的 `lm_eval` 在 `ceval-valid`、`cmmlu`、`arc_easy`、`piqa`、`hellaswag` 上跑分（注意指令模型需加 `--apply_chat_template`）。
2. **量化与跨平台部署（详见 `minimind/README.md` 末尾章节）**：
   - **llama.cpp / GGUF & Ollama**：用 `convert_hf_to_gguf.py` 转为 `FP16` 或 `Q4_K_M`（4-bit 量化后一个 64M 模型只有 **约 40MB**！），编写 `Modelfile` 即可一键 `ollama run minimind`。
   - **MNN 移动端部署**：支持导出 4-bit HQQ 量化模型，在手机 CPU 上离线流畅运行。

### 20.4 前沿架构拓展：当自回归 Transformer 遇上扩散与线性注意力

MiniMind 仓库在 Discussions 中还开源了两个非常值得深入研究的实验性架构分支（均可直接从主线自回归模型权重续训转换而来）：

1. **离散扩散语言模型（Discrete Diffusion Language Model, dLM，[Discussion #618](https://github.com/jingyaogong/minimind/discussions/618)）**：
   - **核心思想**：打破自回归模型「只能从左到右一个词一个词蹦」的限制！把生成过程建模为像图像扩散模型（Diffusion）一样的**并行去噪过程**：初始时整段回复全是 `[MASK]` token，每一步双向注意力同时预测所有 `[MASK]` 位置，先把置信度最高的一部分 token 固定下来，其余继续重掩码（Remasking），只需固定几十步就能**并行生成整段文本**。
   - **从 AR 到 dLM 的桥接**：只需要把 `Attention` 中的单向因果掩码 `is_causal=True` 改为双向注意力（或 Prefix-LM 掩码：Prompt 部分双向可见，回复部分按随机掩码比例 $t \sim U(0,1)$ 遮盖并预测被遮盖位置），就可以直接复用预训练好的 MiniMind 权重微调出扩散语言模型！
2. **线性注意力（Linear Attention，[Discussion #704](https://github.com/jingyaogong/minimind/discussions/704)）**：
   - **核心思想**：标准 Softmax 注意力的计算复杂度是 $O(L^2 d)$，序列长度 $L$ 到几万时二次方开销巨大。线性注意力利用核函数特征映射 $\phi(\cdot)$ 将 $\exp(QK^T)V$ 改写为 $\phi(Q)\big(\phi(K)^T V\big)$——**改变矩阵乘法结合律顺序**，先算 $\mathbf{S}_t = \sum_{j \le t} \phi(k_j)^T v_j \in \mathbb{R}^{d \times d}$！
   - 这样计算复杂度从 $O(L^2 d)$ 直接降为 **$O(L d^2)$（关于序列长度 $L$ 严格线性！）**，并且推理时不再需要随长度增长的 KV Cache，只需要维护一个固定大小的状态矩阵 $\mathbf{S}_t$（就像 RNN 一样恒定显存，却又能像 Transformer 一样并行训练）。

---

## 🛠️ 本阶段动手实验（Labs）

```bash
# 完成 labs/agent.py 中的 TODO(ch19-1) ~ TODO(ch19-2)
python tools/grade.py ch19 -v

# 当所有章节完成后，运行全量闯关评测！
python tools/grade.py
```
