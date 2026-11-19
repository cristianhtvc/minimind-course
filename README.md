# 🧭 Learn-MiniMind：从零手写现代大语言模型全栈课程（架构 · 训练 · 微调 · 蒸馏 · RL · Agent）

> 🌐 **在线交互式课程与 LeetCode 编程主页（支持电脑 · iPad · 手机多端同步）**：
> 👉 **[https://cristianhtvc.github.io/minimind-course/](https://cristianhtvc.github.io/minimind-course/)**
> 📦 **GitHub 仓库**：**[https://github.com/cristianhtvc/minimind-course](https://github.com/cristianhtvc/minimind-course)**

本项目是基于开源项目 **[jingyaogong/minimind](https://github.com/jingyaogong/minimind)**（以及融合 [Datawhale/happy-llm](https://github.com/datawhalechina/happy-llm)、Sebastian Raschka [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) 与 Stanford CS336 的闯关式教学设计）构建的 **互动式 LLM 全栈课程仓库**。

MiniMind 用不到 3000 行原生 PyTorch 代码完整实现了从分词器、Qwen3 风格 Dense/MoE 架构、Pretrain、Full SFT、LoRA、白盒知识蒸馏，到 DPO、PPO、GRPO/CISPO、多轮 Tool-Use Agentic RL 的全套流程。但对于初学者来说，直接扎进 22 个脚本里很容易迷失在分布式训练、混合精度、数据掩码和强化学习公式中。

**本项目将 MiniMind 拆解为「6 大阶段、20 个递进章节、46 个动手编程关卡（TODOs）、15 套自动化对齐测试」**：
- 📖 **每一章先讲透底层原理与数学公式**，再带你**逐行精读 `minimind/` 对应源码与行号**；
- 🧩 **剥离工程噪音的模块化参考实现 `tinymind/`**：与 `MiniMindForCausalLM` **100% 权重兼容、逐元素数值等价**；
- ✍️ **闯关式填空练习 `labs/` + 自动评测器 `tools/grade.py`**：亲手写完核心代码，运行测试立刻验证你的实现是否与 MiniMind 原版完全一致；
- ⚡ **30 秒纯 CPU 全流程微缩实验 `tools/train_tiny_pipeline.py`**：无需 GPU、无需下载数 GB 数据，在本地 30 秒内跑通 `Pretrain -> SFT -> LoRA -> 蒸馏 -> DPO -> GRPO/Agent` 六大阶段！

---

## 🗂️ 仓库目录结构

```text
learn-minimind/
├── index.html                     # 🌐 网页端交互式课程 & 类 LeetCode 在线编程平台（自包含 KaTeX + 11 个可视化沙盒 + 内置评测器）
├── README.md                      # 课程总纲、学习路线图与全模块对照表（本文件）
├── chapters/                      # 📚 7 篇深度教材（底层系统技术栈 + 22 篇顶会论文推导 + MiniMind 逐行源码解析）
│   ├── stage0_underlying_tech_stack_and_papers.md    # ★必读基石：Tensor 内存步幅、IEEE-754 浮点位宽、GPU Roofline 模型、Autograd 与 22 篇论文图谱
│   ├── stage1_ch00_03_foundations_and_tokenizer.md   # 第 0-3 章：全景地图、LM 数学基础、Byte-Level BPE 与 ChatML 模板
│   ├── stage2_ch04_07_transformer_components.md      # 第 4-7 章：RMSNorm、GQA、QK-Norm、KV Cache、RoPE/YaRN、SwiGLU
│   ├── stage2_ch08_10_model_moe_and_generation.md    # 第 8-10 章：64M 参数精算、MoE 路由与辅助损失、五大采样解码策略
│   ├── stage3_ch11_13_pretrain_and_sft.md            # 第 11-13 章：Pretrain/SFT 数据掩码本质差异、现代训练循环工程
│   ├── stage4_ch14_15_lora_and_distillation.md       # 第 14-15 章：70 行手写 LoRA 与合并、黑盒 vs 白盒知识蒸馏
│   ├── stage5_ch16_18_rlhf_dpo_ppo_grpo.md           # 第 16-18 章：PO 统一视角、DPO、PPO(GAE)、GRPO/CISPO 与 SGLang 引擎
│   └── stage6_ch19_20_agent_and_deployment.md        # 第 19-20 章：多轮 Tool-Call Agentic RL、格式转换、API 部署与 dLM/线性注意力
├── tinymind/                      # ✅ 模块化、详细中文注释的参考实现（与 MiniMind 权重互载 & 数值等价）
├── labs/                          # ✍️ 你的动手练习区（46 个核心算法被挖空为 TODO，等待你亲手实现）
├── tests/                         # 🧪 15 套自动化单元测试（检验形状、梯度以及与 MiniMind 原版的逐元素等价性）
├── tools/
│   ├── web_ide.py                 # 🖥️ 启动网页端 LeetCode 实时 PyTorch 评测服务（http://127.0.0.1:8765）
│   ├── build_web_course.py        # 🏗️ 生成并校验自包含交互式网页 index.html（含 510 个 LaTeX 公式零错误自检）
│   ├── grade.py                   # 🏆 命令行闯关评测器（按章节检查 labs/ 进度）
│   ├── train_tiny_pipeline.py     # ⚡ 30 秒 CPU 全流程微缩训练实验（一次跑通 6 大训练阶段）
│   └── make_labs.py               # 🔧 从 tinymind/ 自动生成 labs/ 练习骨架
└── minimind/                      # 📦 上游 jingyaogong/minimind 完整源码（供随时对照跳转阅读）
```

---

## 🚀 3 分钟快速上手

### 1. 运行 30 秒全流程微缩实验（直观感受 LLM 训练全生命周期）

```bash
cd learn-minimind
../.venv/bin/python tools/train_tiny_pipeline.py
```

你将亲眼看到一个约 53 万参数的微缩版 `TinyMind`（使用 MiniMind 真实的 6400 词表分词器）在几秒内依次完成：
1. **Pretrain**：交叉熵损失从理论随机猜测值 $\ln(6400) \approx 8.76$ 下降到 `1.83`；
2. **Full SFT**：仅对 Assistant 回复计算损失，Loss 降到 `0.02`，实际对话测试输出 `"你是谁？" -> "我是MiniMind小助手。"`；
3. **LoRA**：仅训练 `0.38%` 的参数完成领域微调，并将增量矩阵 $BA$ 无损合并回主干权重（误差 $< 10^{-5}$）；
4. **White-Box Distillation**：用 4 层 Teacher 的软分布（$T=2.0$）指导 2 层 Student；
5. **DPO**：Chosen 与 Rejected 回答的对数概率差（Margin）翻倍拉大；
6. **GRPO / CISPO & Agent Tool Call**：解析 `<tool_call>`、安全计算算术表达式、根据 Ground Truth 打分并算出组内相对优势 $A_i$。

### 2. 开始你的闯关式学习

```bash
# 查看当前所有章节的闯关进度（默认评测 labs/ 目录）
../.venv/bin/python tools/grade.py

# 阅读第一阶段教程，然后打开 labs/basics.py 完成 TODO(ch01-1) ~ TODO(ch02-2)，再单独评测该关：
../.venv/bin/python tools/grade.py ch01 -v

# 如果想验证 tinymind/ 参考答案（15/15 全部通过）：
../.venv/bin/python tools/grade.py --ref
```

---

## 🗺️ 20 章全栈学习路线与「知识 ↔ MiniMind 源码 ↔ 动手实验」总对照表

```
[阶段一: 基础与分词] ──▶ [阶段二: 模型架构与生成] ──▶ [阶段三: 预训练与SFT]
   Ch00 全景地图            Ch04 RMSNorm & PreNorm      Ch11 PretrainDataset
   Ch01-02 交叉熵/PPL       Ch05 GQA & KV Cache         Ch12 训练循环工程
   Ch03 Byte-Level BPE     Ch06 RoPE & YaRN 外推        Ch13 SFT 只算回复Loss
                           Ch07 SwiGLU FFN                   │
                           Ch08 64M模型组装                   ▼
                           Ch09 MoE 混合专家          [阶段四: 轻量化与蒸馏]
                           Ch10 五大采样解码策略         Ch14 70行手写LoRA
                                                       Ch15 黑盒/白盒知识蒸馏
                                                             │
[阶段六: Agent与部署] ◀── [阶段五: 强化学习对齐] ◀──────────────┘
   Ch19 多轮Agentic RL      Ch16 PO统一视角与DPO
   Ch20 转换/评测/部署/dLM   Ch17 PPO 与 GAE 优势估计
                           Ch18 GRPO/CISPO 与 SGLang
```

| 阶段 | 章节 | 你应该掌握的核心理论与拓展知识 | 对应 MiniMind 源码位置（点击精确跳转） | 参考实现 (`tinymind/`) | 动手练习 (`labs/` TODOs) | 闯关评测命令 |
| :---: | :--- | :--- | :--- | :--- | :--- | :--- |
| **一** | **Ch00**<br>[导览](chapters/stage1_ch00_03_foundations_and_tokenizer.md) | 22 个文件的依赖关系、大模型全生命周期 5 大模块划分 | [minimind/README.md](minimind/README.md) | — | — | — |
| **一** | **Ch01–02**<br>[LM 数学基础](chapters/stage1_ch00_03_foundations_and_tokenizer.md) | 自回归条件概率链、数值稳定 Softmax、LogSumExp、`ignore_index=-100`、PPL vs BPB（跨分词器指标） | [model_minimind.py#L255-L258](minimind/model/model_minimind.py#L255-L258) | [basics.py](tinymind/basics.py) | `TODO(ch01-1)`<br>`TODO(ch02-1..2)` | `python tools/grade.py ch01` |
| **一** | **Ch03**<br>[BPE 分词器](chapters/stage1_ch00_03_foundations_and_tokenizer.md) | 为什么选 6400 词表（Embedding 参数账本）、Byte-Level BPE 训练与优先级编码、中文 UTF-8 流式解码 `\ufffd` 缓冲、Jinja2 `chat_template` 与 `<think>` 开关 | [train_tokenizer.py#L45-L186](minimind/trainer/train_tokenizer.py#L45-L186)<br>[tokenizer_config.json](minimind/model/tokenizer_config.json) | [tokenizer_bpe.py](tinymind/tokenizer_bpe.py) | `TODO(ch03-1..4)` | `python tools/grade.py ch03` |
| **二** | **Ch04**<br>[RMSNorm](chapters/stage2_ch04_07_transformer_components.md) | LayerNorm vs RMSNorm、为什么混合精度下要先 `.float()` 再 `.type_as(x)`、Post-Norm vs Pre-Norm 梯度高速公路 | [model_minimind.py#L50-L60](minimind/model/model_minimind.py#L50-L60) | [norm.py](tinymind/norm.py) | `TODO(ch04-1)` | `python tools/grade.py ch04` |
| **二** | **Ch05**<br>[GQA & KV Cache](chapters/stage2_ch04_07_transformer_components.md) | MHA / MQA / GQA 显存权衡、`repeat_kv` 零拷贝广播、Qwen3 风格 `q_norm/k_norm`、Prefill vs Decode KV Cache 拼接、因果掩码 `scores[:, :, :, -seq_len:]` | [model_minimind.py#L86-L134](minimind/model/model_minimind.py#L86-L134) | [attention.py](tinymind/attention.py) | `TODO(ch05-1..4)` | `python tools/grade.py ch05` |
| **二** | **Ch06**<br>[RoPE & YaRN](chapters/stage2_ch04_07_transformer_components.md) | 复平面旋转为什么能让内积只依赖相对距离 $m-n$、`rotate_half` 分半配对、YaRN 高低频分治与线性斜坡 `ramp` 外推 32K 上下文 | [model_minimind.py#L31-L39](minimind/model/model_minimind.py#L31-L39)<br>[model_minimind.py#L62-L84](minimind/model/model_minimind.py#L62-L84) | [rope.py](tinymind/rope.py) | `TODO(ch06-1..5)` | `python tools/grade.py ch06` |
| **二** | **Ch07**<br>[SwiGLU FFN](chapters/stage2_ch04_07_transformer_components.md) | 从 ReLU FFN 到 SwiGLU 三矩阵门控（`gate_proj`, `up_proj`, `down_proj`）、为什么中间维度向上对齐 64 倍数 | [model_minimind.py#L26](minimind/model/model_minimind.py#L26)<br>[model_minimind.py#L136-L146](minimind/model/model_minimind.py#L136-L146) | [ffn.py](tinymind/ffn.py) | `TODO(ch07-1)` | `python tools/grade.py ch07` |
| **二** | **Ch08**<br>[完整模型组装](chapters/stage2_ch08_10_model_moe_and_generation.md) | `Block -> Model -> ForCausalLM` 三层封装、词嵌入与输出层 Weight Tying、`logits_to_keep` 切片、63.91M 参数逐层手算表、MobileLLM 深窄 vs 矮胖 Scaling Law | [model_minimind.py#L183-L258](minimind/model/model_minimind.py#L183-L258) | [model.py](tinymind/model.py) | `TODO(ch08-1..3)` | `python tools/grade.py ch08` |
| **二** | **Ch09**<br>[MoE 混合专家](chapters/stage2_ch08_10_model_moe_and_generation.md) | 198M 总参数 vs 64M 激活参数、Top-K 路由、$k=1$ 时的 STE 直通梯度估计 `top1 - top1.detach() + 1`、DDP 零乘哑梯度、Switch 负载均衡损失 `aux_loss` | [model_minimind.py#L148-L181](minimind/model/model_minimind.py#L148-L181) | [moe.py](tinymind/moe.py) | `TODO(ch09-1..3)` | `python tools/grade.py ch08` |
| **二** | **Ch10**<br>[自回归生成](chapters/stage2_ch08_10_model_moe_and_generation.md) | Temperature、Repetition Penalty 正负不对称缩放、Top-K、Top-P 核采样右移一位防全 `-inf`、Batch `finished` 状态机 | [model_minimind.py#L261-L293](minimind/model/model_minimind.py#L261-L293) | [generate.py](tinymind/generate.py) | `TODO(ch10-1..4)` | `python tools/grade.py ch10` |
| **三** | **Ch11**<br>[预训练数据](chapters/stage3_ch11_13_pretrain_and_sft.md) | `PretrainDataset` 构造、Padding 置 `-100`、大规模语料 MinHash/SimHash 去重与 `max_seq_len` 权衡 | [lm_dataset.py#L40-L58](minimind/dataset/lm_dataset.py#L40-L58) | [data.py](tinymind/data.py) | `TODO(ch11-1)` | `python tools/grade.py ch11` |
| **三** | **Ch12**<br>[训练循环工程](chapters/stage3_ch11_13_pretrain_and_sft.md) | 余弦退火 `get_lr`、`bfloat16` vs `float16`+`GradScaler`、梯度累积除 `accumulation_steps`、梯度裁剪、`.tmp` 原子保存与跨卡数续训 `SkipBatchSampler` | [train_pretrain.py](minimind/trainer/train_pretrain.py)<br>[trainer_utils.py#L42-L159](minimind/trainer/trainer_utils.py#L42-L159) | [train_utils.py](tinymind/train_utils.py) | `TODO(ch12-1..3)` | `python tools/grade.py ch12` |
| **三** | **Ch13**<br>[Full SFT](chapters/stage3_ch11_13_pretrain_and_sft.md) | Pretrain 与 SFT 的唯一本质差异：子序列匹配只保留 `<\|im_start\|>assistant\n` 到 `<\|im_end\|>\n` 的 label、20% 随机 System 与 80% 移除空 `<think>` | [lm_dataset.py#L9-L39](minimind/dataset/lm_dataset.py#L9-L39)<br>[lm_dataset.py#L61-L122](minimind/dataset/lm_dataset.py#L61-L122)<br>[train_full_sft.py](minimind/trainer/train_full_sft.py) | [data.py](tinymind/data.py) | `TODO(ch13-1)` | `python tools/grade.py ch11` |
| **四** | **Ch14**<br>[手写 LoRA](chapters/stage4_ch14_15_lora_and_distillation.md) | 内在秩假说与 $\Delta W = \frac{\alpha}{r}BA$、为什么 $A$ 高斯初始化而 $B$ 必须全 0 初始化、Python 闭包默认参数显式绑定、零开销权重合并 `merge_lora` | [model_lora.py](minimind/model/model_lora.py)<br>[train_lora.py](minimind/trainer/train_lora.py) | [lora.py](tinymind/lora.py) | `TODO(ch14-1..3)` | `python tools/grade.py ch14` |
| **四** | **Ch15**<br>[知识蒸馏 KD](chapters/stage4_ch14_15_lora_and_distillation.md) | 黑盒蒸馏（数据合成）vs 白盒蒸馏（Logits KL）、暗知识与软化温度 $T$、为什么 KL 损失必须乘以 $T^2$（梯度一阶泰勒推导） | [train_distillation.py#L24-L98](minimind/trainer/train_distillation.py#L24-L98) | [distill.py](tinymind/distill.py) | `TODO(ch15-1)` | `python tools/grade.py ch15` |
| **五** | **Ch16**<br>[PO 统一视角 & DPO](chapters/stage5_ch16_18_rlhf_dpo_ppo_grpo.md) | 策略项 $\Phi(r_t, A_t)$ + 优势项 $A_t$ + KL 正则项统一公式、从 Bradley-Terry 推导 DPO 解析损失、`chosen/rejected` 拼接 `2B` 单次前向 | [lm_dataset.py#L125-L196](minimind/dataset/lm_dataset.py#L125-L196)<br>[train_dpo.py#L25-L84](minimind/trainer/train_dpo.py#L25-L84) | [dpo.py](tinymind/dpo.py) | `TODO(ch16-1..2)` | `python tools/grade.py ch16` |
| **五** | **Ch17**<br>[PPO 与 GAE](chapters/stage5_ch16_18_rlhf_dpo_ppo_grpo.md) | Actor/Critic/Ref/Reward 四模型架构、`CriticModel` 价值头、末尾放置稀疏奖励、GAE 倒序递推、Policy+Value 双裁剪、k3 非负 KL 估计器 $e^\Delta - \Delta - 1$、DDP 早停同步 | [train_ppo.py#L36-L245](minimind/trainer/train_ppo.py#L36-L245) | [ppo.py](tinymind/ppo.py) | `TODO(ch17-1..3)` | `python tools/grade.py ch17` |
| **五** | **Ch18**<br>[GRPO / CISPO](chapters/stage5_ch16_18_rlhf_dpo_ppo_grpo.md) | 为什么组内相对奖励 $A_i = (R_i - \mu_G)/\sigma_G$ 能省掉 Critic、EOS 截断掩码、CISPO 裁剪重要性权重保留稀缺反思词梯度、SGLang 在线权重热更新 | [train_grpo.py#L31-L194](minimind/trainer/train_grpo.py#L31-L194)<br>[rollout_engine.py](minimind/trainer/rollout_engine.py) | [grpo.py](tinymind/grpo.py) | `TODO(ch18-1..2)` | `python tools/grade.py ch17` |
| **六** | **Ch19**<br>[Agentic RL](chapters/stage6_ch19_20_agent_and_deployment.md) | `<tools>` 与 `<tool_call>` 协议、多轮交互 Rollout 中「模型生成 token `mask=1` vs 环境返回 `<tool_response>` `mask=0`」、AST 安全求值 `safe_math_eval`、可验证奖励 RLVR | [train_agent.py#L33-L355](minimind/trainer/train_agent.py#L33-L355)<br>[trainer_utils.py#L182-L208](minimind/trainer/trainer_utils.py#L182-L208)<br>[eval_toolcall.py](minimind/scripts/eval_toolcall.py) | [agent.py](tinymind/agent.py) | `TODO(ch19-1..2)` | `python tools/grade.py ch19` |
| **六** | **Ch20**<br>[部署与前沿拓展](chapters/stage6_ch19_20_agent_and_deployment.md) | `.pth` 与 HuggingFace `Llama/Qwen3` 格式双向转换、OpenAI API 服务端、C-Eval 评测、vLLM/Ollama/GGUF/MNN 量化部署、离散扩散语言模型（dLM）与线性注意力 | [convert_model.py](minimind/scripts/convert_model.py)<br>[serve_openai_api.py](minimind/scripts/serve_openai_api.py)<br>[web_demo.py](minimind/scripts/web_demo.py) | — | — | — |

---

## 🌐 部署为你的 GitHub Pages 个人学习主页（支持电脑 · iPad · 手机多端同步）

本仓库已内置完整的 GitHub Pages 静态主页支持（[`index.html`](index.html)、[`.nojekyll`](.nojekyll)、PWA [`manifest.json`](manifest.json)、GitHub Actions 自动测试与部署工作流 [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml)，以及一键推送脚本 [`tools/publish_github.sh`](tools/publish_github.sh)）。

### 1. 一键上传并开启 GitHub Pages

1. 登录 GitHub，点击右上角 **New repository** 新建一个公开（Public）仓库：
   - **想作为个人根主页**（访问地址为 `https://<你的用户名>.github.io/`）：将仓库命名为 **`<你的用户名>.github.io`**；
   - **想作为独立子主页**（访问地址为 `https://<你的用户名>.github.io/learn-minimind/`）：将仓库命名为 **`learn-minimind`**。
2. 在本地终端运行一键发布脚本：
   ```bash
   cd learn-minimind

   # 方式 A：推送到 <你的用户名>.github.io 个人根主页
   bash tools/publish_github.sh <你的GitHub用户名>

   # 方式 B：推送到 learn-minimind 项目子主页
   bash tools/publish_github.sh <你的GitHub用户名> learn-minimind
   ```
3. 打开 GitHub 仓库的 **Settings → Pages**：
   - 在 **Build and deployment → Source** 中选择 **`GitHub Actions`**（或选择 `Deploy from a branch` → `main` / `/(root)`）。
   - 等待约 30 秒，即可在电脑、iPad、手机浏览器上随时访问你的专属 LLM 交互式学习主页！
   - 📱 **手机 / iPad 全屏 App 模式**：在 iOS Safari 或 Android Chrome 打开你的主页后，点击「分享 → 添加到主屏幕（Add to Home Screen）」，即可以原生全屏 App 形态运行。

### 2. 多设备端学习进度与代码同步（3 种方式）

点击网页顶部导航栏的 **`☁️ 多端同步`** 按钮，可展开多端同步中心：
- **① 免登录跨设备同步链接（`#sync=...`）**：在电脑上点击「🔗 复制跨设备同步链接」，会把你当前的通关状态与所有自定义代码打包编码进 URL Hash 中，发送到手机或 iPad 打开链接即刻无损恢复进度。
- **② 本地 JSON 存档导出 / 导入**：一键将 18 个章节的全部代码修改与通关记录导出为 `.json` 文件，在任意设备上导入。
- **③ GitHub Gist 云端实时同步**：在网页中填入具有 `gist` 权限的 GitHub Personal Access Token（仅存储在你当前浏览器的 `localStorage` 中），即可通过「☁️ 上传到云端」与「🔄 从云端拉取」在多台设备间秒级同步所有代码与闯关进度。

---

## 💡 给初学者的学习建议

1. **先跑通微缩流水线，再进单章**：先执行一次 `python tools/train_tiny_pipeline.py`，建立对 6 个训练阶段输入输出的感性认识。
2. **边读教程、边看对照源码、边填 `labs/`**：每读完一章 `chapters/stage*.md`，打开对应的 `labs/xxx.py`，找到 `# TODO(chXX-Y)`，根据提示自己写几行 PyTorch 代码，然后运行 `python tools/grade.py chXX -v`。如果卡住超过 15 分钟，随时对照同名的 `tinymind/xxx.py` 参考答案。
3. **如果你有单卡 GPU（哪怕是一张笔记本 RTX 3060/4060 或租一张 1.3 元/小时的 RTX 3090）**：当你通关全部 15 个单元测试后，进入 `minimind/` 目录，下载 `pretrain_t2t_mini.jsonl` 与 `sft_t2t_mini.jsonl`，亲手跑一遍真刀真枪的 64M 模型训练——那时 `minimind` 里的每一行代码对你来说都将是透明的！
