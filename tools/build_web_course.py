"""
构建 Learn-MiniMind 网页端交互式课程与 LeetCode 在线编程平台 (learn-minimind/index.html)。

设计要点：
  1. 数学公式零故障保障：
     所有 LaTeX 公式均通过 Python 原始字符串 r"..." 配合 M() / MB() 辅助函数写入 HTML 的 `data-tex` 属性，
     彻底杜绝 Markdown 下划线 `_`、尖括号 `<` 或 JS 转义符 `\f \t \b \r \n` 破坏公式。
     内联打包 KaTeX 0.16.11，页面加载时以 `throwOnError: true` 严格渲染并做自检计数。
  2. 底层技术栈 + 22 篇核心论文 + 逐行源码对照 + 交互式可视化沙盒：
     每个章节均包含「底层系统与数学第一性原理推导」、「奠基与前沿论文卡片」、「交互式动态可视化实验台」、「MiniMind 源码逐行对照」。
  3. 类 LeetCode 网页端在线编译器与双模评测：
     右侧内置代码编辑器（预置 labs/*.py 的真实练习代码）、提示、答案对照与测试用例控制台；
     既支持纯网页离线评测（即开即用，无需启动任何服务），也支持连接 `python tools/web_ide.py` 调用本地 PyTorch 内核做真实张量评测。
"""
import base64
import html
import json
import re
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABS_DIR = ROOT / "labs"
TINYMIND_DIR = ROOT / "tinymind"
ASSETS_DIR = ROOT / "web" / "assets"


def M(tex: str) -> str:
    """行内 LaTeX 公式标签：将原始 TeX 安全存入 data-tex 属性，由 KaTeX 直接渲染。"""
    return f'<span class="math-inline" data-tex="{html.escape(tex, quote=True)}"></span>'


def MB(tex: str) -> str:
    """块级独立 LaTeX 公式标签。"""
    return f'<div class="math-block" data-tex="{html.escape(tex, quote=True)}"></div>'


def T(raw_html: str) -> str:
    # 先替换块级 $$...$$，再替换行内 $...$
    s = re.sub(r"\$\$([\s\S]+?)\$\$", lambda m: MB(m.group(1).strip()), raw_html)
    s = re.sub(r"\$([^\$\n]+?)\$", lambda m: M(m.group(1).strip()), s)
    return s


def read_code_pair(filename: str):
    lab_p = LABS_DIR / filename
    ref_p = TINYMIND_DIR / filename
    return (
        lab_p.read_text(encoding="utf-8") if lab_p.exists() else "",
        ref_p.read_text(encoding="utf-8") if ref_p.exists() else "",
    )


def load_guides(chapters):
    guides = json.loads((ROOT / "web/lesson_guides.json").read_text(encoding="utf-8"))
    if guides.keys() != {ch["id"] for ch in chapters}:
        raise ValueError("章节目录与导读目录不一致")
    for ch in chapters:
        guide = guides[ch["id"]]
        path, marker, count, symbol, note = guide["source"]
        lines = (ROOT / "minimind" / path).read_text(encoding="utf-8").splitlines()
        matches = [i for i, line in enumerate(lines) if line.lstrip().startswith(marker)]
        if len(matches) != 1 or matches[0] + count > len(lines):
            raise ValueError(f'{ch["id"]} 源码摘录无法定位到 {path} 中的 {marker!r}')
        start = matches[0]
        guide["source"] = {
            "path": path, "start": start + 1, "end": start + count, "symbol": symbol, "note": note,
            "code": textwrap.dedent("\n".join(lines[start:start + count])),
        }
        for target, _ in guide["uses"]:
            if target not in guides and not (ROOT / "minimind" / target).is_file():
                raise ValueError(f'{ch["id"]} 的应用入口不存在: {target}')
        activity = guide["practice"][0]
        if activity != "vis" and f"TODO({activity})" not in ch["todos"]:
            raise ValueError(f'{ch["id"]} 的练习入口不存在: {activity}')
        ch["guide"] = guide


def build_chapters_data():
    chapters = [
        # ===================================================================== 0. 底层基石
        {
            "id": "ch00",
            "stage": "阶段 0 · 底层基石",
            "badge": "必读基石",
            "difficulty": "Foundation",
            "title": "第 0 章：LLM 四大底层技术栈（内存步幅 · 浮点位宽 · GPU Roofline · Autograd）",
            "subtitle": "在写第一行模型代码前，先打通 PyTorch 物理内存、IEEE-754 精度、GPU 存储层次与自动微分计算图",
            "file": "basics.py",
            "mm_path": "minimind/model/model_minimind.py (全局底层机制)",
            "visualizer": "fp_roofline",
            "todos": ["TODO(ch01-1)", "TODO(ch02-1)", "TODO(ch02-2)"],
            "theory_html": T(r"""
<h4>1. 步幅决定怎样读一块内存</h4>
<p>把六个数排成两行 <code>[[0, 1, 2], [3, 4, 5]]</code>。沿行走一步要跨过 3 个数，沿列走一步只跨 1 个数，步幅就是 <code>(3, 1)</code>。转置后，第一行读成 <code>[0, 3]</code>，步幅换成 <code>(1, 3)</code>，底层六个数仍按原顺序存放。</p>
$$\text{storage\_index}(i_0,i_1,i_2,i_3)=\text{offset}+\sum_{k=0}^3 i_k s_k$$
<p><strong>用在注意力的哪里？</strong> 第 5 章把张量从「批次、序列、头、特征」转成「批次、头、序列、特征」，让每个头分别算注意力。转置往往只改步幅。若新的维度无法按原步幅合并，<code>view</code> 会报错；<code>reshape</code> 可以在需要时复制。<code>repeat_kv</code> 中的 <code>expand</code> 通过零步幅共享数据，之后的 <code>reshape</code> 仍可能分配内存。</p>

<h4>2. 浮点精度决定哪些中间结果存得下</h4>
<p>在二进制正规浮点数里，符号位决定正负，指数决定数量级，尾数决定这一数量级里能区分多细的差异。</p>
$$x=(-1)^S 2^{E-\mathrm{bias}}\left(1+\sum_{k=1}^{M}b_k2^{-k}\right)$$
<ul>
  <li><strong>FP16</strong> 用 5 位指数和 10 位尾数，最大有限值为 65,504。平方后的数可能上溢，反向传播中的小梯度也可能被舍入到零。</li>
  <li><strong>BF16</strong> 用 8 位指数和 7 位尾数，有接近 FP32 的取值范围，相邻可表示数之间的间隔更大。</li>
  <li><strong>用在归一化与训练</strong> 第 4 章的 RMSNorm 用 FP32（8 位指数、23 位尾数）计算平方均值，再恢复输入精度。第 12 章的 FP16 训练还会放大损失，在更新参数前还原梯度尺度。</li>
</ul>

<h4>3. 用带宽估算一次解码要等多久</h4>
<p>线性层把输入向量乘以权重矩阵。权重从显存读进来以后，同一批中的多个 token 可以复用它。一次只处理一个 token 时，同样一批权重只贡献一行输出。用「运算次数除以读写字节数」可以估算哪项资源先到上限。</p>
$$\mathcal I=\frac{\mathrm{FLOPs}}{\mathrm{Bytes}},\qquad \mathcal I_{\mathrm{ridge}}=\frac{P_{\mathrm{peak}}}{\mathrm{BW}}$$
<p>例如以 312 TFLOP/s 和 1.5 TB/s 作估算，拐点约为 208 FLOP/byte。若主要流量来自 BF16 权重，一次读 2 字节并做一次乘加（2 FLOP），单 token、单样本时约为 1 FLOP/byte。实际耗时还取决于缓存、批量和算子实现。</p>
<p><strong>用在生成的哪里？</strong> 第 10 章先一次读完整段提示（prefill），再逐词解码（decode）。第 5 章的 KV 缓存省去重算历史键值的工作，GQA 还减少需要保存和读取的键值头数。</p>

<h4>4. Autograd 怎样把一次预测变成一次权重更新？</h4>
<p>让一个标量参数 $w=2$ 乘以输入 $x=3$，目标是 5。预测为 6，平方损失为 1。反向传播沿「平方 → 相减 → 相乘」走回来，把每一步的局部导数相乘。</p>
$$L=(wx-5)^2,\qquad \frac{\partial L}{\partial w}=2(wx-5)x=6$$
<p>以学习率 0.01 更新一次，$w$ 变成 1.94，预测变成 5.82，误差缩小。<strong>去第 12 章</strong>对照 <code>loss.backward()</code> 与 <code>optimizer.step()</code> 的分工。去第 9 章再看 <code>detach()</code>，它让某一条计算路径在反向时停止传递梯度。</p>
"""),
            "papers": [
                {
                    "title": "Mixed Precision Training",
                    "authors": "Paulius Micikevicius et al. (NVIDIA & Baidu)",
                    "venue": "ICLR 2018 · arXiv:1710.03740",
                    "pain": "直接用 FP16 训练深度神经网络会因为指数位太窄（仅 5 位）导致激活平方上溢与微小权重梯度下溢为 0，模型无法收敛。",
                    "formula": r"W_{\text{master}}^{(t+1)} = W_{\text{master}}^{(t)} - \eta \cdot \frac{1}{S} \nabla_{W_{\text{fp16}}} \big(S \cdot \mathcal{L}(W_{\text{fp16}})\big)",
                    "minimind": "对应 trainer/train_pretrain.py 中的 torch.cuda.amp.autocast(dtype=dtype) 与 GradScaler(enabled=(args.dtype == 'float16'))，以及 RMSNorm / Attention 中先 .float() 规约再转回半精度的设计。"
                },
                {
                    "title": "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
                    "authors": "Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré",
                    "venue": "NeurIPS 2022 · arXiv:2205.14135",
                    "pain": "标准 Self-Attention 需要在慢速 HBM 显存中反复读写 N×N 的注意力分数矩阵 S 和概率矩阵 P，显存占用 O(N²) 且被 HBM 带宽卡死。",
                    "formula": r"m(\mathbf{x}) = \max_i x_i,\quad \ell(\mathbf{x}) = \sum_i e^{x_i - m(\mathbf{x})},\quad \ell_{\text{new}} = e^{m_1 - m_{\text{new}}}\ell_1 + e^{m_2 - m_{\text{new}}}\ell_2",
                    "minimind": "对应 model/model_minimind.py 第 109, 125-126 行的 F.scaled_dot_product_attention，在 SRAM 内通过 Online Softmax 分块融合计算。"
                }
            ],
            "code_walkthrough": [
                ("model/model_minimind.py:L56-L60", "def norm(self, x): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)\ndef forward(self, x): return (self.weight * self.norm(x.float())).type_as(x)", "先转 float32 计算平方均值与倒数平方根 rsqrt，防止 FP16 上溢与 BF16 累加精度损失，再转回输入 dtype。"),
                ("model/model_minimind.py:L86-L89", "def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:\n    bs, slen, num_key_value_heads, head_dim = x.shape\n    if n_rep == 1: return x\n    return x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim)", "利用 [:, :, :, None, :] 插入步幅为 0 的广播维度 (.expand 零拷贝)，再用 .reshape 合并头维度。"),
                ("model/model_minimind.py:L165-L166", "top1 = torch.topk(F.softmax(self.gate(x_flat.detach()), dim=-1), k=1, dim=-1, sorted=False)[0]\ntopk_weight = top1 - top1.detach() + 1.0", "直通梯度估计器 (STE)：前向值恒为 1.0，反向传播时梯度跳过 .detach() 直接回传给路由器 self.gate。")
            ],
            "hints": [
                "本章与第 1-2 章共享练习文件 basics.py：在 stable_softmax 中，先减去 logits.max(dim=dim, keepdim=True).values 再求 exp()。",
                "在 cross_entropy 中，使用 logits - torch.logsumexp(logits, dim=-1, keepdim=True) 得到数值稳定的 log_probs，并用 targets != ignore_index 过滤有效位置。",
                "perplexity(mean_ce_loss) 即 math.exp(mean_ce_loss)。"
            ]
        },

        # ===================================================================== 1. 第 1-2 章
        {
            "id": "ch01-02",
            "stage": "阶段 1 · 基础与分词",
            "badge": "核心必做",
            "difficulty": "Easy",
            "title": "第 1–2 章：自回归语言模型数学主线（Softmax · LogSumExp 交叉熵 · 困惑度）",
            "subtitle": "从概率链式法则推导交叉熵损失对 Logits 的优美梯度 p - y，亲手实现防溢出 Softmax 与 Cross-Entropy",
            "file": "basics.py",
            "mm_path": "minimind/model/model_minimind.py#L254-L258",
            "visualizer": "softmax_ce",
            "todos": ["TODO(ch01-1)", "TODO(ch02-1)", "TODO(ch02-2)"],
            "theory_html": T(r"""
<h4>1. 自回归语言模型的第一性原理：条件概率链式法则</h4>
<p>给定一段长度为 $T$ 的离散 token 序列 $\mathbf{x} = (x_1, x_2, \dots, x_T)$，其中每个 $x_t \in \{0, 1, \dots, V-1\}$（MiniMind 中词表大小 $V = 6400$）。根据概率论的乘法链式法则，任意联合概率分布都可以无损分解为条件概率的连乘：</p>
$$P_\theta(x_1, x_2, \dots, x_T) = \prod_{t=1}^{T} P_\theta(x_t \mid x_1, x_2, \dots, x_{t-1})$$
<p>训练语言模型的目标，就是在语料库上做<strong>最大似然估计（Maximum Likelihood Estimation, MLE）</strong>，等价于最小化每个位置的<strong>负对数似然（Negative Log-Likelihood, NLL）</strong>：</p>
$$\mathcal{L}(\theta) = -\frac{1}{T} \sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t})$$

<h4>2. 数值稳定 Softmax 与 LogSumExp 恒等式推导</h4>
<p>神经网络最后一层线性头 <code>lm_head</code> 在位置 $t$ 输出未归一化的实数打分向量（Logits）$\mathbf{z} = (z_1, \dots, z_V) \in \mathbb{R}^V$。通过 Softmax 映射为概率分布：</p>
$$p_i = \operatorname{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^V e^{z_j}} = \frac{e^{z_i - M}}{\sum_{j=1}^V e^{z_j - M}}, \qquad M = \max_{1 \le k \le V} z_k$$
<p>由于 $z_j - M \le 0$，每一项 $e^{z_j - M} \in (0, 1]$ <strong>永远不会发生指数上溢</strong>，且分母中至少有一项为 $e^{M-M} = 1$，因此分母 $\ge 1$ <strong>永远不会下溢为 0</strong>。进一步对 $p_i$ 取对数，得到计算交叉熵时必用的 <strong>LogSumExp 公式</strong>：</p>
$$\log p_i = z_i - \operatorname{LSE}(\mathbf{z}), \qquad \operatorname{LSE}(\mathbf{z}) = M + \log \sum_{j=1}^{V} e^{z_j - M}$$

<h4>3. 为什么交叉熵配 Softmax 的梯度如此优美？（完整求导）</h4>
<p>设真实目标 token 为 $y$（对应的独热向量为 $\mathbf{y}$，其中 $y_k = \mathbb{I}(k = y)$），单步交叉熵损失为 $\ell = -\log p_y = -z_y + \log \sum_{j=1}^V e^{z_j}$。对任意第 $i$ 个 logit $z_i$ 求偏导：</p>
$$\frac{\partial \ell}{\partial z_i} = -\mathbb{I}(i = y) + \frac{\partial}{\partial z_i} \log \sum_{j=1}^V e^{z_j} = -y_i + \frac{e^{z_i}}{\sum_{j=1}^V e^{z_j}} = p_i - y_i$$
<p><strong>物理直觉</strong>：损失对 Logits 的梯度就等于<strong>「模型预测概率 $\mathbf{p}$ 减去真实 One-Hot 标签 $\mathbf{y}$」</strong>！如果模型预测严重错误（$p_y \to 0$），梯度 $p_y - 1 \to -1$ 幅度最大、丝毫不饱和；随着 $p_y \to 1$，梯度平滑趋近于 0。</p>

<h4>4. 困惑度（PPL）与跨词表公平指标 BPB（Bits Per Byte）</h4>
<ul>
  <li><strong>随机初始化时的理论 Loss</strong>：刚初始化时 $z_i \approx 0$，预测接近均匀分布 $p_i = 1/V$，因此理论初始损失为 $\mathcal{L}_0 = -\log(1/6400) = \ln 6400 \approx \mathbf{8.764}$，对应困惑度 $\text{PPL}_0 = e^{\mathcal{L}_0} = 6400$（每步如同在 6400 个等可能词中盲猜）。</li>
  <li><strong>BPB（每字节比特数）</strong>：不同分词器切出的 token 数 $N_{\text{tok}}$ 不同，无法直接对比按 token 平均的 PPL。将总信息量（以 2 为底的比特数）除以原始文本的 UTF-8 字节数 $N_{\text{bytes}}$，即可得到与分词器无关的绝对压缩率指标：</li>
</ul>
$$\text{PPL} = \exp(\mathcal{L}_{\text{CE}}), \qquad \text{BPB} = \frac{N_{\text{tok}}}{N_{\text{bytes}}} \cdot \frac{\mathcal{L}_{\text{CE}}}{\ln 2}$$
"""),
            "papers": [
                {
                    "title": "A Neural Probabilistic Language Model",
                    "authors": "Yoshua Bengio, Réjean Ducharme, Pascal Vincent, Christian Jauvin",
                    "venue": "JMLR 2003 (Vol. 3, pp. 1137–1155)",
                    "pain": "传统统计 N-gram 语言模型基于离散词频计数，遇到训练集中未出现过的词序列时概率为 0，参数量随上下文窗口呈指数级爆炸 V^n。",
                    "formula": r"\hat{P}(w_t \mid w_{t-n+1}^{t-1}) = \frac{\exp(y_{w_t})}{\sum_{i=1}^{V} \exp(y_i)}, \quad \mathbf{y} = \mathbf{b} + W \mathbf{x} + U \tanh(\mathbf{d} + H \mathbf{x})",
                    "minimind": "奠定了「词嵌入查表 embed_tokens -> 神经网络特征变换 -> 输出层线性投影 lm_head -> Softmax 交叉熵」的现代神经语言模型范式。"
                },
                {
                    "title": "Improving Language Understanding by Generative Pre-Training (GPT-1)",
                    "authors": "Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever (OpenAI)",
                    "venue": "OpenAI Technical Report 2018",
                    "pain": "NLP 各下游任务依赖大量人工标注数据，缺乏通用的无监督表征学习框架。",
                    "formula": r"\mathcal{L}_1(\mathcal{U}) = \sum_{i} \log P(u_i \mid u_{i-k}, \dots, u_{i-1}; \Theta)",
                    "minimind": "对应 model/model_minimind.py 第 255-257 行中 logits[..., :-1, :] 与 labels[..., 1:] 的错位自回归交叉熵训练目标。"
                }
            ],
            "code_walkthrough": [
                ("model/model_minimind.py:L255-L257", "if labels is not None:\n    x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()\n    loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)", "位置 0..L-2 的 logits 预测位置 1..L-1 的 labels；展平为 [B*(L-1), V] 后计算交叉熵，自动跳过 label == -100 的位置。")
            ],
            "hints": [
                "TODO(ch01-1) stable_softmax：z = logits - logits.max(dim=dim, keepdim=True).values; e = z.exp(); return e / e.sum(dim=dim, keepdim=True)",
                "TODO(ch02-1) cross_entropy：先筛出 valid = targets != ignore_index，再用 log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)，最后用 -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1).mean()。",
                "TODO(ch02-2) perplexity：直接返回 math.exp(mean_ce_loss)。"
            ]
        },

        # ===================================================================== 2. 第 3 章
        {
            "id": "ch03",
            "stage": "阶段 1 · 基础与分词",
            "badge": "核心必做",
            "difficulty": "Medium",
            "title": "第 3 章：Byte-Level BPE 分词器算法与 ChatML / Jinja2 模板机制",
            "subtitle": "亲手从 256 个 UTF-8 基础字节出发训练 BPE 词表，彻底搞懂中文多字节流式缓冲与 <think> 模板开关",
            "file": "tokenizer_bpe.py",
            "mm_path": "minimind/trainer/train_tokenizer.py#L45-L186",
            "visualizer": "bpe_vis",
            "todos": ["TODO(ch03-1)", "TODO(ch03-2)", "TODO(ch03-3)", "TODO(ch03-4)"],
            "theory_html": T(r"""
<h4>1. 词表大小的参数账本：为什么 MiniMind 选 $V = 6400$ 而不是 15 万？</h4>
<p>词嵌入矩阵 <code>embed_tokens.weight</code> 的形状为 $[V, d_{\text{model}}]$。在 MiniMind 默认配置 $d_{\text{model}} = 768$ 下：</p>
$$N_{\text{embed}}(V) = V \times d_{\text{model}} = \begin{cases} 151,643 \times 768 \approx \mathbf{116.46\text{M}} & (\text{若用 Qwen 词表，比 8 层主干 59M 大一倍！}) \\ 6,400 \times 768 \approx \mathbf{4.92\text{M}} & (\text{MiniMind 词表，仅占总参数 } 63.9\text{M 的 } 7.7\%) \end{cases}$$
<p>对于百兆以内的小模型，精简词表并开启 Weight Tying（与 <code>lm_head</code> 共享同一块矩阵内存），能把 92% 以上的参数预算留给真正负责上下文推理的 Transformer 层。</p>

<h4>2. Byte-Level BPE 的严格形式化算法</h4>
<p>设初始词表为 256 个基础字节 $\mathcal{V}_0 = \{0, 1, \dots, 255\}$。任意文本经正则预切分后，转化为字节序列集合 $\mathcal{D}_0 = \{\mathbf{u}^{(1)}, \dots, \mathbf{u}^{(M)}\}$。</p>
<ul>
  <li><strong>训练阶段（共执行 $K = V - 256 - |\mathcal{S}|$ 轮贪心合并）</strong>：在第 $k$ 轮（$k = 0, \dots, K-1$），统计所有片段内相邻符号对 $(a, b)$ 的共现频次，选取最高频对合并为新 token ID $v_{\text{new}} = 256 + k$：</li>
</ul>
$$(a_k^*, b_k^*) = \underset{(a, b) \in \mathcal{V}_k \times \mathcal{V}_k}{\operatorname{arg\,max}} \sum_{\mathbf{u} \in \mathcal{D}_k} \sum_{t=1}^{|\mathbf{u}|-1} \mathbb{I}(u_t = a \wedge u_{t+1} = b)$$
<ul>
  <li><strong>编码阶段（按优先级 Rank 合并）</strong>：给定新单词的字节序列 $\mathbf{u}$，<strong>绝不能按当前出现次数合并，而必须按训练时学到的合并顺序（即 <code>merges[pair]</code> 最小的 pair 优先）</strong>循环替换，直到没有任何相邻对存在于 <code>merges</code> 字典中：</li>
</ul>
$$p^* = \underset{p \in \text{adjacent\_pairs}(\mathbf{u})}{\operatorname{arg\,min}} \operatorname{rank}(p), \qquad \operatorname{rank}(p) = \begin{cases} \text{merges}[p] & p \in \text{merges} \\ +\infty & p \notin \text{merges} \end{cases}$$

<h4>3. UTF-8 多字节编码与流式解码的「半个汉字」陷阱</h4>
<p>UTF-8 是变长字节编码：ASCII 占 1 字节（<code>0xxxxxxx</code>），而绝大多数中文汉字占 <strong>3 个字节</strong>（格式为 <code>1110xxxx 10xxxxxx 10xxxxxx</code>）。当一个生僻汉字被拆成 2 个 token（如前 2 字节是一个 token，第 3 字节是下一个 token）时，如果在模型刚吐出第 1 个 token 时立即调用 <code>bytes.decode('utf-8', errors='replace')</code>，不完整的 UTF-8 字节流会解码出 Unicode 替换符 <code>\ufffd</code>（显示为 ）。因此流式输出时必须维护 <code>token_cache</code>，<strong>只有当解码结果不含 <code>\ufffd</code> 时才真正输出并清空缓冲</strong>。</p>
"""),
            "papers": [
                {
                    "title": "Neural Machine Translation of Rare Words with Subword Units (BPE)",
                    "authors": "Rico Sennrich, Barry Haddow, Alexandra Birch",
                    "venue": "ACL 2016 · arXiv:1508.07909",
                    "pain": "固定词典无法处理罕见词、复合词和屈折变化（如你看/看着/看见，play/playing/played），传统方案只能回退到 <unk> 或逐字符建模。",
                    "formula": r"\mathcal{V}_{k+1} = \mathcal{V}_k \cup \{a_k^* \circ b_k^*\}, \quad (a_k^*, b_k^*) = \arg\max_{(a,b)} \operatorname{count}(a, b)",
                    "minimind": "对应 tinymind/tokenizer_bpe.py 中的 get_stats、merge、ByteLevelBPE.train 与 _encode_chunk。"
                },
                {
                    "title": "Language Models are Unsupervised Multitask Learners (GPT-2 Byte-Level BPE)",
                    "authors": "Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever",
                    "venue": "OpenAI 2019",
                    "pain": "如果直接以 Unicode 字符作为 BPE 的初始词表，仅基础字符就超过 13 万个；若只保留常见字符则仍会出现 <unk>。",
                    "formula": r"|\mathcal{V}_{\text{base}}| = 256 \text{ (UTF-8 Bytes)} \implies \text{OOV Rate} \equiv 0\%",
                    "minimind": "对应 trainer/train_tokenizer.py 第 47, 67 行的 pre_tokenizers.ByteLevel(add_prefix_space=False) 与 decoders.ByteLevel()。"
                }
            ],
            "code_walkthrough": [
                ("trainer/train_tokenizer.py:L61-L69", "num_buffer = special_tokens_num - len(special_tokens_list + additional_tokens_list)\nbuffer_tokens = [f'<|buffer{i}|>' for i in range(1, num_buffer + 1)]\nall_special_tokens = special_tokens_list + additional_tokens_list + buffer_tokens\ntrainer = trainers.BpeTrainer(vocab_size=vocab_size, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=all_special_tokens)", "预留 36 个特殊 Token 槽位（0: <|endoftext|>, 1: <|im_start|>, 2: <|im_end|> 等），加 256 个基础字节，其余 6108 个槽位由语料 BPE 合并产生。"),
                ("trainer/train_tokenizer.py:L81-L84", "for token_info in tokenizer_data.get('added_tokens', []):\n    if token_info['content'] not in special_tokens_list:\n        token_info['special'] = False", "关键细节：将 <think>、</think>、<tool_call>、</tool_call> 的 special 置为 False，防止 decode(skip_special_tokens=True) 把思考标签和工具标签吞掉！"),
                ("trainer/train_tokenizer.py:L177-L185", "for tid in input_ids:\n    token_cache.append(tid)\n    current_decode = tokenizer.decode(token_cache)\n    if current_decode and '\\ufffd' not in current_decode:\n        ...\n        token_cache = []", "流式字节缓冲：当一个汉字的 3 个 UTF-8 字节被拆到两个 token 中时，检测到 \\ufffd 暂不输出，直到下一个 token 拼齐完整字符。")
            ],
            "hints": [
                "TODO(ch03-1) get_stats：遍历 for pair in zip(ids, ids[1:]): counts[pair] = counts.get(pair, 0) + 1",
                "TODO(ch03-2) merge：用 while i < len(ids) 扫描，当 i < len(ids)-1 且 (ids[i], ids[i+1]) == pair 时 append(idx) 并 i += 2，否则 append(ids[i]) 并 i += 1。",
                "TODO(ch03-3) train：每轮对 ids_list 中每个 chunk 调 get_stats(ids, stats)，取 pair = max(stats, key=stats.get)，new_id = 256 + i，对所有 chunk 执行 merge，并记录 self.merges[pair] = new_id 和 self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]。",
                "TODO(ch03-4) _encode_chunk：stats = get_stats(ids)，找优先级最高的 pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))，若不在 self.merges 中则 break，否则 ids = merge(ids, pair, self.merges[pair])。"
            ]
        },

        # ===================================================================== 3. 第 4 章
        {
            "id": "ch04",
            "stage": "阶段 2 · 模型架构",
            "badge": "核心必做",
            "difficulty": "Easy",
            "title": "第 4 章：RMSNorm 与 Pre-Norm 残差梯度高速公路",
            "subtitle": "从雅可比矩阵链式法则严格证明为什么 Pre-Norm 不会梯度消失，实现与 MiniMind 完全等价的混合精度 RMSNorm",
            "file": "norm.py",
            "mm_path": "minimind/model/model_minimind.py#L50-L60",
            "visualizer": "norm_grad",
            "todos": ["TODO(ch04-1)"],
            "theory_html": T(r"""
<h4>1. LayerNorm vs RMSNorm：为什么去掉均值中心化不仅不掉点还能提速？</h4>
<p>给定输入向量 $\mathbf{x} \in \mathbb{R}^d$，经典 LayerNorm 与 RMSNorm（Zhang & Sennrich, 2019）的定义对比如下：</p>
$$\operatorname{LayerNorm}(\mathbf{x})_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma_i + \beta_i, \qquad \mu = \frac{1}{d}\sum_{j=1}^d x_j,\quad \sigma^2 = \frac{1}{d}\sum_{j=1}^d (x_j - \mu)^2$$
$$\operatorname{RMSNorm}(\mathbf{x})_i = \frac{x_i}{\operatorname{RMS}(\mathbf{x})} \cdot \gamma_i = x_i \cdot \left(\frac{1}{d}\sum_{j=1}^d x_j^2 + \epsilon\right)^{-1/2} \cdot \gamma_i$$
<ul>
  <li><strong>缩放不变性（Scale Invariance）</strong>：对于任意常数 $\alpha > 0$，有 $\operatorname{RMSNorm}(\alpha \mathbf{x}) = \operatorname{RMSNorm}(\mathbf{x})$。这保证了特征与梯度范数不会随网络层数呈指数爆炸或衰减。</li>
  <li><strong>硬件收益</strong>：LayerNorm 需要两次遍历向量（第一次求均值 $\mu$，第二次求方差 $\sigma^2$），而激活归一化是典型的 <strong>Memory-Bound 算子</strong>。RMSNorm 只需一次规约求平方均值 $\frac{1}{d}\sum x_j^2$ 并调用硬件倒数平方根指令 <code>torch.rsqrt</code>，访存减少约 40%，算子端到端提速 10%~30%。</li>
</ul>

<h4>2. Pre-Norm vs Post-Norm 的反向传播梯度流严格推导</h4>
<p>对比深度为 $L$ 层的两种 Transformer 残差连接方式：</p>
<ul>
  <li><strong>Post-Norm（2017 原版 Transformer）</strong>：$\mathbf{x}_{l+1} = \operatorname{Norm}(\mathbf{x}_l + F_l(\mathbf{x}_l))$。每一层主干都被包裹在非线性的 <code>Norm</code> 内。设第 $l$ 层归一化处的输入标准差为 $\sigma_l$，由於 $\operatorname{Var}(\mathbf{x}_l + F_l(\mathbf{x}_l)) > \operatorname{Var}(\mathbf{x}_l)$，反向传播经过每一层 <code>Norm</code> 时梯度都会被乘以一个小于 1 的缩放因子 $1/\sigma_l$，连乘 $L$ 层后浅层梯度呈指数级衰减 $\prod_{l=1}^L (1/\sigma_l) \to 0$！</li>
  <li><strong>Pre-Norm（MiniMind / LLaMA / Qwen3 标配）</strong>：将 <code>RMSNorm</code> 移入残差支路内部：</li>
</ul>
$$\mathbf{x}_{l+1} = \mathbf{x}_l + F_l(\operatorname{RMSNorm}(\mathbf{x}_l)) \implies \mathbf{x}_L = \mathbf{x}_0 + \sum_{l=0}^{L-1} F_l(\operatorname{RMSNorm}(\mathbf{x}_l))$$
<p>对第 0 层输入 $\mathbf{x}_0$ 求损失 $\mathcal{L}$ 的梯度（链式法则）：</p>
$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}_0} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}_L} \cdot \left( \mathbf{I} + \sum_{l=0}^{L-1} \frac{\partial F_l(\operatorname{RMSNorm}(\mathbf{x}_l))}{\partial \mathbf{x}_0} \right)$$
<p>看到括号里那个<strong>恒等矩阵 $\mathbf{I}$</strong> 了吗？无论网络有 8 层还是 100 层，顶层的梯度 $\frac{\partial \mathcal{L}}{\partial \mathbf{x}_L}$ 都能通过 $\mathbf{I}$ 无损直达最底层 $\mathbf{x}_0$！这就是 <strong>Pre-Norm 残差梯度高速公路</strong>。</p>
"""),
            "papers": [
                {
                    "title": "Root Mean Square Layer Normalization (RMSNorm)",
                    "authors": "Biao Zhang, Rico Sennrich",
                    "venue": "NeurIPS 2019 · arXiv:1910.07467",
                    "pain": "LayerNorm 在每个时间步和每一层都要做均值和方差两次向量规约，在序列模型中占用了显著的计算和显存带宽时间。",
                    "formula": r"\bar{a}_i = \frac{a_i}{\operatorname{RMS}(\mathbf{a})} g_i, \quad \operatorname{RMS}(\mathbf{a}) = \sqrt{\frac{1}{n} \sum_{i=1}^{n} a_i^2}",
                    "minimind": "对应 model/model_minimind.py 第 50-60 行的 RMSNorm 类。"
                },
                {
                    "title": "On Layer Normalization in the Transformer Architecture",
                    "authors": "Ruibin Xiong et al.",
                    "venue": "ICML 2020 · arXiv:2002.04745",
                    "pain": "原版 Post-LN Transformer 必须依赖数千步的 Learning Rate Warmup，一旦去掉或缩短 Warmup 就会在训练初期发散崩溃。",
                    "formula": r"\mathbb{E}\left[\left\|\frac{\partial \mathcal{L}}{\partial \mathbf{x}_l^{\text{Pre-LN}}}\right\|_2^2\right] = \mathcal{O}(1) \quad \text{vs.} \quad \mathbb{E}\left[\left\|\frac{\partial \mathcal{L}}{\partial \mathbf{x}_l^{\text{Post-LN}}}\right\|_2^2\right] = \mathcal{O}\left(\left(\frac{2}{3}\right)^{L-l}\right)",
                    "minimind": "对应 model/model_minimind.py 第 192-198 行的 MiniMindBlock.forward：先 input_layernorm 再 self_attn 后加回 residual。"
                }
            ],
            "code_walkthrough": [
                ("model/model_minimind.py:L50-L60", "class RMSNorm(torch.nn.Module):\n    def __init__(self, dim: int, eps: float = 1e-5):\n        super().__init__()\n        self.eps = eps\n        self.weight = nn.Parameter(torch.ones(dim))\n    def norm(self, x):\n        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)\n    def forward(self, x):\n        return (self.weight * self.norm(x.float())).type_as(x)", "torch.rsqrt 比 1.0 / torch.sqrt 更快（直接映射为 GPU SFU 硬件指令 rsqrt.approx）；keepdim=True 保证形状为 [..., 1] 以便与 x 逐元素广播。")
            ],
            "hints": [
                "TODO(ch04-1) RMSNorm.norm：一行即可 return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)"
            ]
        },

        # ===================================================================== 4. 第 5 章
        {
            "id": "ch05",
            "stage": "阶段 2 · 模型架构",
            "badge": "核心必做",
            "difficulty": "Hard",
            "title": "第 5 章：分组查询注意力 (GQA)、QK-Norm、因果掩码与 KV Cache",
            "subtitle": "推导 KV Cache 显存公式与带历史缓存时的非方阵因果掩码切片，手写完整现代 Attention 层",
            "file": "attention.py",
            "mm_path": "minimind/model/model_minimind.py#L86-L134",
            "visualizer": "gqa_kv",
            "todos": ["TODO(ch05-1)", "TODO(ch05-2)", "TODO(ch05-3)", "TODO(ch05-4)"],
            "theory_html": T(r"""
<h4>1. 缩放点积注意力：为什么要除以 $\sqrt{d_k}$？</h4>
<p>假设 Query 和 Key 的每个分量独立同分布，满足零均值、单位方差 $\mathbb{E}[q_i]=\mathbb{E}[k_i]=0,\; \operatorname{Var}(q_i)=\operatorname{Var}(k_i)=1$。那么它们的点积 $S = \mathbf{q} \cdot \mathbf{k} = \sum_{i=1}^{d_k} q_i k_i$ 的均值与方差为：</p>
$$\mathbb{E}[S] = 0, \qquad \operatorname{Var}(S) = \sum_{i=1}^{d_k} \operatorname{Var}(q_i k_i) = d_k \implies \operatorname{Std}(S) = \sqrt{d_k}$$
<p>当 $d_k = 96$ 时，点积的标准差高达 $\sqrt{96} \approx 9.8$，大量元素绝对值超过 20，导致 Softmax 进入极度尖锐的饱和区（最大项概率接近 1，其余全为 0，雅可比梯度 $\operatorname{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^T \approx \mathbf{0}$）。除以 $\sqrt{d_k}$ 正是为了把点积的方差<strong>严格拉回 1</strong>！</p>

<h4>2. Qwen3 风格的 QK-Norm：防止训练后期的注意力熵坍塌</h4>
<p>即便除以了 $\sqrt{d_k}$，随着训练推进，投影矩阵 $W_Q, W_K$ 的范数仍可能持续膨胀，导致 $\operatorname{Var}(q_i) \gg 1$。MiniMind-3 对齐 Qwen3，在分头之后、施加 RoPE 之前，对每个头的 $\mathbf{q}, \mathbf{k} \in \mathbb{R}^{d_k}$ 独立做一次 <code>RMSNorm(head_dim)</code>：</p>
$$\tilde{\mathbf{q}}_h = \operatorname{RMSNorm}(\mathbf{q}_h), \quad \tilde{\mathbf{k}}_h = \operatorname{RMSNorm}(\mathbf{k}_h) \implies \frac{\|\tilde{\mathbf{q}}_h\|_2}{\sqrt{d_k}} \approx \|\boldsymbol{\gamma}_q\|_2$$
<p>这把注意力 Logits 的上界牢牢锁死，使小模型也能耐受大得多的学习率。</p>

<h4>3. GQA 与 KV Cache 显存账本</h4>
<p>设批大小为 $B$，序列长度为 $L$，层数为 $N_{\text{layers}}$，Query 头数为 $H_q$，KV 头数为 $H_{kv}$，每头维度 $d_h$，每个浮点数占 2 字节（BF16/FP16）。则缓存全部历史 K 和 V 所需的显存为：</p>
$$\text{Memory}_{\text{KV}} = 2_{\text{(K和V)}} \times 2_{\text{(Bytes)}} \times B \times N_{\text{layers}} \times L \times H_{kv} \times d_h = 4 B N_{\text{layers}} L H_{kv} d_h \text{ (Bytes)}$$
<p>当 $H_{kv} = H_q / n_{\text{rep}}$（MiniMind 中 $H_q=8, H_{kv}=4, n_{\text{rep}}=2$）时，KV Cache 显存与 Decode 阶段每步的 HBM 访存带宽直接降低为 MHA 的 $1 / n_{\text{rep}} = 50\%$！</p>

<h4>4. 带 KV Cache 时的非方阵因果掩码：为什么是 <code>scores[:, :, :, -seq_len:]</code>？</h4>
<p>当存在历史缓存长度 $L_{\text{past}} > 0$，且当前新输入长度为 $L_{\text{cur}} = \texttt{seq\_len}$ 时：</p>
<ul>
  <li>当前 Query 形状为 $[B, H_q, L_{\text{cur}}, d_h]$；拼接历史缓存后的 Key 形状为 $[B, H_q, L_{\text{past}} + L_{\text{cur}}, d_h]$。</li>
  <li>因此注意力分数矩阵 <code>scores</code> 是<strong>长方形矩阵</strong>，形状为 $[B, H_q, L_{\text{cur}}, L_{\text{past}} + L_{\text{cur}}]$！</li>
  <li>前 $L_{\text{past}}$ 列全部是<strong>已经发生过的历史 token</strong>，当前这 $L_{\text{cur}}$ 个 token 对它们全部可见（掩码为 0）；只有<strong>最后 $L_{\text{cur}}$ 列（即 <code>scores[:, :, :, -seq_len:]</code>）</strong>是当前输入的 token 互相观看，需要加上 $L_{\text{cur}} \times L_{\text{cur}}$ 的上三角 $-\infty$ 掩码（<code>triu(1)</code>）！</li>
</ul>
"""),
            "papers": [
                {
                    "title": "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints",
                    "authors": "Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, Sumit Sanghai (Google)",
                    "venue": "EMNLP 2023 · arXiv:2305.13245",
                    "pain": "MHA 在长序列推理时 KV Cache 显存和带宽开销过大；而极端压缩到 1 个 KV 头的 MQA 会造成推理质量下降和训练不稳定。",
                    "formula": r"\text{Group } g \in \{1,\dots,G\}: \quad \{\mathbf{q}_{g,1}, \dots, \mathbf{q}_{g, H_q/G}\} \text{ share single } (\mathbf{k}_g, \mathbf{v}_g)",
                    "minimind": "对应 model/model_minimind.py 第 22-23, 86-124 行的 num_attention_heads=8, num_key_value_heads=4 与 repeat_kv。"
                },
                {
                    "title": "Scaling Vision Transformers to 22 Billion Parameters / Qwen3 Technical Report (QK-Norm)",
                    "authors": "Mostafa Dehghani et al. (Google Brain 2023) / Qwen Team (2025)",
                    "venue": "ICML 2023 · arXiv:2302.05442",
                    "pain": "高学习率或深层网络训练中，Q 和 K 的内积随训练步数增长失控，导致注意力权重坍缩成 One-Hot，梯度消失。",
                    "formula": r"\mathbf{A} = \operatorname{softmax}\left(\frac{\operatorname{RMSNorm}(\mathbf{Q}) \operatorname{RMSNorm}(\mathbf{K})^T}{\sqrt{d_k}}\right)",
                    "minimind": "对应 model/model_minimind.py 第 104-105, 117 行的 self.q_norm = RMSNorm(self.head_dim) 与 xq, xk = self.q_norm(xq), self.k_norm(xk)。"
                }
            ],
            "code_walkthrough": [
                ("model/model_minimind.py:L113-L119", "xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)\nxq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)\nxk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)\nxv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)\nxq, xk = self.q_norm(xq), self.k_norm(xk)\ncos, sin = position_embeddings\nxq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)", "注意严格顺序：线性投影 -> view 拆多头 -> QK-Norm -> 施加 RoPE 旋转位置编码 -> 拼接 KV Cache。"),
                ("model/model_minimind.py:L128-L131", "scores = (xq @ xk.transpose(-2, -1)).float() / math.sqrt(self.head_dim)\nif self.is_causal: scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float('-inf'), device=scores.device).triu(1)\nif attention_mask is not None: scores += (1.0 - attention_mask.to(scores.dtype).unsqueeze(1).unsqueeze(2)) * -1e9\noutput = self.attn_dropout(F.softmax(scores, dim=-1).type_as(xq)) @ xv", "手写注意力路径：转 float32 算缩放点积 -> 仅对最后 seq_len 列加因果上三角 -inf -> 加 padding 掩码 -> softmax 后转回原精度乘以 xv。")
            ],
            "hints": [
                "TODO(ch05-1) repeat_kv：return x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim)",
                "TODO(ch05-2) Q/K/V 投影与 QK-Norm：分别通过 q_proj/k_proj/v_proj，view 为 4D 张量后执行 xq, xk = self.q_norm(xq), self.k_norm(xk)。",
                "TODO(ch05-3) KV Cache：if past_key_value is not None: 在 dim=1 上 torch.cat([past_key_value[0], xk], dim=1)（xv 同理）；past_kv = (xk, xv) if use_cache else None。",
                "TODO(ch05-4) 手写注意力：完全按照上方源码解析中的 4 行实现即可。"
            ]
        },

        # ===================================================================== 5. 第 6 章
        {
            "id": "ch06",
            "stage": "阶段 2 · 模型架构",
            "badge": "核心必做",
            "difficulty": "Hard",
            "title": "第 6 章：RoPE 旋转位置编码与 YaRN 长上下文外推",
            "subtitle": "从复平面欧拉公式推导旋转不变内积，彻底搞懂 NTK-by-parts / YaRN 高低频分治斜坡函数",
            "file": "rope.py",
            "mm_path": "minimind/model/model_minimind.py#L62-L84",
            "visualizer": "rope_yarn",
            "todos": ["TODO(ch06-1)", "TODO(ch06-2)", "TODO(ch06-3)", "TODO(ch06-4)", "TODO(ch06-5)"],
            "theory_html": T(r"""
<h4>1. RoPE 的第一性原理：如何用「绝对位置旋转」产生「相对位置内积」？</h4>
<p>我们希望寻找一个位置编码变换 $f(\mathbf{v}, m)$，使得位置 $m$ 的 Query 与位置 $n$ 的 Key 的点积<strong>只依赖于词向量本身和它们的相对距离 $m - n$</strong>：</p>
$$\langle f(\mathbf{q}, m),\, f(\mathbf{k}, n) \rangle = g(\mathbf{q}, \mathbf{k}, m - n)$$
<p>将二维向量 $\mathbf{q} = (q_1, q_2)^T$ 看作复平面上的复数 $q = q_1 + i q_2$。根据欧拉公式 $e^{im\theta} = \cos(m\theta) + i\sin(m\theta)$，将复数乘以 $e^{im\theta}$ 等价于在平面上逆时针旋转角度 $m\theta$：</p>
$$f(\mathbf{q}, m) = R_{m\theta} \mathbf{q} = \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix} \begin{pmatrix} q_1 \\ q_2 \end{pmatrix} = \begin{pmatrix} q_1 \cos m\theta - q_2 \sin m\theta \\ q_2 \cos m\theta + q_1 \sin m\theta \end{pmatrix}$$
<p>由于旋转矩阵是正交矩阵（满足 $R_\alpha^T R_\beta = R_{\beta - \alpha}$），位置 $m$ 的 Query 与位置 $n$ 的 Key 的内积为：</p>
$$\langle R_{m\theta}\mathbf{q},\, R_{n\theta}\mathbf{k} \rangle = \mathbf{q}^T R_{m\theta}^T R_{n\theta} \mathbf{k} = \mathbf{q}^T R_{(n-m)\theta} \mathbf{k} = \operatorname{Re}\!\left(q \bar{k} e^{i(m-n)\theta}\right)$$
<p><strong>结论惊人地简洁</strong>：每个 token 在进入 Attention 时只需要知道自己的绝对位置 $m$（完美兼容 KV Cache，历史 token 缓存后无需随新 token 到来重新编码！），而点积结果却天然只取决于相对距离 $m - n$！</p>

<h4>2. 多维频率表与 <code>rotate_half</code> 分半配对</h4>
<p>将 $d$ 维头向量分成 $d/2$ 对，第 $i \in \{0, \dots, d/2-1\}$ 对使用几何级数递减频率（波长 $\lambda_i = 2\pi / \theta_i$ 从 $2\pi$ 到 $2\pi \cdot \text{base}$）：</p>
$$\theta_i = \frac{1}{\text{base}^{2i/d}}, \qquad \text{MiniMind 中 } \text{base} = \texttt{rope\_theta} = 10^6$$
<p>MiniMind 将前半向量 $\mathbf{x}_{0 : d/2}$ 与后半向量 $\mathbf{x}_{d/2 : d}$ 配对，定义 $\operatorname{rotate\_half}([\mathbf{x}_1, \mathbf{x}_2]) = [-\mathbf{x}_2, \mathbf{x}_1]$，于是二维旋转公式直接化为向量逐元素乘加：</p>
$$\operatorname{RoPE}(\mathbf{x}, m) = \mathbf{x} \odot \cos(m\boldsymbol{\theta}) + \operatorname{rotate\_half}(\mathbf{x}) \odot \sin(m\boldsymbol{\theta})$$

<h4>3. YaRN 长上下文外推公式推导（为什么不能所有维度一起除以 $s$？）</h4>
<p>设预训练最大长度为 $L_{\text{orig}} = 2048$，推理目标长度为 $L_{\text{ext}} = 32768$（扩展倍数 $s = 16$）。第 $i$ 维在预训练长度内转过的总圈数为 $r(i) = \frac{L_{\text{orig}}}{\lambda_i} = \frac{L_{\text{orig}} \theta_i}{2\pi}$。反解出「恰好转 $b$ 圈」对应的维度下标 $d(b)$：</p>
$$r(i) = b \iff \frac{L_{\text{orig}}}{2\pi \cdot \text{base}^{2i/d}} = b \iff i = d(b) = \frac{d \cdot \ln\big(L_{\text{orig}} / (2\pi b)\big)}{2 \ln(\text{base})}$$
<ul>
  <li><strong>高频维（$i < d(\beta_{\text{fast}}),\, \beta_{\text{fast}}=32$）</strong>：训练时已转了 32 圈以上，负责局部相邻词序，<strong>完全不缩放（$\gamma_i = 0$）</strong>。</li>
  <li><strong>低频维（$i > d(\beta_{\text{slow}}),\, \beta_{\text{slow}}=1$）</strong>：训练时连 1 圈都没转完，外推会遇到陌生角度，<strong>做线性位置插值（$\gamma_i = 1$，频率除以 $s$）</strong>。</li>
  <li><strong>中频维</strong>：用线性斜坡 $\gamma_i = \operatorname{clamp}\!\left(\frac{i - i_{\text{low}}}{i_{\text{high}} - i_{\text{low}}}, 0, 1\right)$ 平滑混合：</li>
</ul>
$$\theta_i^{\text{YaRN}} = \theta_i \cdot \left( (1 - \gamma_i) + \frac{\gamma_i}{s} \right)$$
"""),
            "papers": [
                {
                    "title": "RoFormer: Enhanced Transformer with Rotary Position Embedding",
                    "authors": "Jianlin Su (苏剑林), Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu",
                    "venue": "arXiv:2104.09864 (2021) / Neurocomputing 2024",
                    "pain": "加性绝对位置编码缺乏相对距离偏置且长度外推性差；而把相对位置直接加到注意力矩阵上又破坏了内积结构且不利于 KV 缓存。",
                    "formula": r"f_{\{q,k\}}(\mathbf{x}_m, m) = \mathbf{R}_{\Theta, m}^d \mathbf{W}_{\{q,k\}} \mathbf{x}_m",
                    "minimind": "对应 model/model_minimind.py 第 62-84 行的 precompute_freqs_cis 与 apply_rotary_pos_emb。"
                },
                {
                    "title": "YaRN: Efficient Context Window Extension of Large Language Models",
                    "authors": "Bowen Peng, Jeffrey Quesnelle, Honglu Fan, Enrico Shippole",
                    "venue": "ICLR 2024 · arXiv:2309.00071",
                    "pain": "直接线性位置插值 (PI, θ' = θ/s) 把高频维度也压缩了 s 倍，导致相邻 token 的旋转角差距太小，模型丧失局部精细词序感知能力。",
                    "formula": r"h(\theta_d) = (1 - \gamma(r(d))) \theta_d + \gamma(r(d)) \frac{\theta_d}{s}",
                    "minimind": "对应 model/model_minimind.py 第 31-39, 64-73 行的 inference_rope_scaling (YaRN) 实现。"
                }
            ],
            "code_walkthrough": [
                ("model/model_minimind.py:L70-L73", "inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))\nlow, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)\nramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)\nfreqs = freqs * (1 - ramp + ramp / factor)", "仅用 4 行就实现了 YaRN 的高低频分界维度反解、线性斜坡 ramp 与频率混合缩放。"),
                ("model/model_minimind.py:L74-L78", "t = torch.arange(end, device=freqs.device)\nfreqs = torch.outer(t, freqs).float()\nfreqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor\nfreqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor", "外积 torch.outer(t, freqs) 生成 [end, dim/2] 角度表，再在最后一维拼接两份变成 [end, dim]，与 rotate_half 的前后分半配对严格对应。")
            ],
            "hints": [
                "TODO(ch06-1)：freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))",
                "TODO(ch06-2)：ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1); freqs = freqs * (1 - ramp + ramp / factor)",
                "TODO(ch06-3)：t = torch.arange(end, device=freqs.device); freqs = torch.outer(t, freqs).float(); 再用 torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor 构造 freqs_cos 和 freqs_sin。",
                "TODO(ch06-4) rotate_half：return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)",
                "TODO(ch06-5) apply_rotary_pos_emb：q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)（k 同理）。"
            ]
        },

        # ===================================================================== 6. 第 7 章
        {
            "id": "ch07",
            "stage": "阶段 2 · 模型架构",
            "badge": "核心必做",
            "difficulty": "Easy",
            "title": "第 7 章：SwiGLU 门控前馈网络与 Tensor Core 64 倍数维度对齐",
            "subtitle": "理解为什么现代 LLM 把两层 ReLU FFN 换成了三矩阵 SwiGLU，以及 SiLU 自门控的平滑梯度性质",
            "file": "ffn.py",
            "mm_path": "minimind/model/model_minimind.py#L136-L146",
            "visualizer": "param_calc",
            "todos": ["TODO(ch07-1)"],
            "theory_html": T(r"""
<h4>1. 从 ReLU FFN 到 GLU 再到 SwiGLU</h4>
<p>在 Transformer 中，Attention 负责<strong>跨 token 的信息路由与聚合</strong>，而逐位置前馈网络（Position-wise FFN）负责对每个 token 的特征做<strong>非线性变换与知识记忆存储</strong>（研究表明 FFN 相当于键值记忆库，占整层参数的 76%）。</p>
<ul>
  <li><strong>2017 原版 ReLU FFN</strong>：$\operatorname{FFN}(\mathbf{x}) = \max(0, \mathbf{x}W_1 + \mathbf{b}_1)W_2 + \mathbf{b}_2$。ReLU 在负半轴导数恒为 0，容易发生「神经元死亡（Dead Neurons）」。</li>
  <li><strong>SiLU / Swish 激活函数</strong>：$\operatorname{SiLU}(z) = z \cdot \sigma(z) = \frac{z}{1 + e^{-z}}$。处处光滑可导，且在负半轴有微小的非零负响应，缓解梯度截断。</li>
  <li><strong>SwiGLU（Shazeer, 2020）</strong>：引入两路并行的升维投影 <code>gate_proj</code> ($W_g$) 与 <code>up_proj</code> ($W_u$)，其中一路经 SiLU 激活作为<strong>软门控（Soft Gate）</strong>控制另一路的信息流，再由 <code>down_proj</code> ($W_d$) 降维回 $d_{\text{model}}$：</li>
</ul>
$$\operatorname{SwiGLU}(\mathbf{x}) = W_d \Big( \operatorname{SiLU}(W_g \mathbf{x}) \odot (W_u \mathbf{x}) \Big)$$
<p>从微积分角度看，双线性门控乘积 $\operatorname{SiLU}(W_g\mathbf{x}) \odot (W_u\mathbf{x})$ 在反向传播时提供了<strong>两路互补的梯度通道</strong>（乘积求导法则 $(uv)' = u'v + uv'$），显著提升了特征交互的表达阶数。</p>

<h4>2. 为什么 <code>intermediate_size</code> 选 $\lceil d \cdot \pi / 64 \rceil \times 64$？</h4>
<p>传统两矩阵 FFN 中间维取 $4d$ 时参数量为 $2 \times d \times 4d = 8d^2$。SwiGLU 有三个矩阵（$W_g, W_u, W_d$），参数量为 $3 \times d \times d_{\text{ff}}$。当 $d = 768$ 时，MiniMind 取：</p>
$$d_{\text{ff}} = \left\lceil \frac{768 \times \pi}{64} \right\rceil \times 64 = \lceil 37.699 \rceil \times 64 = 38 \times 64 = 2432$$
<p>向上取整到 <strong>64 的整数倍</strong>，是为了让矩阵维度严格对齐 GPU Tensor Core 的 $16 \times 16$ / $64 \times 64$ 硬件矩阵乘法 Tile 以及 128-Byte 显存缓存行。</p>
"""),
            "papers": [
                {
                    "title": "GLU Variants Improve Transformer (SwiGLU)",
                    "authors": "Noam Shazeer (Google Brain)",
                    "venue": "arXiv:2002.05202 (2020)",
                    "pain": "标准 FFN 的单路激活函数在同等参数与算力预算下表征能力尚未饱和。",
                    "formula": r"\operatorname{FFN}_{\text{SwiGLU}}(\mathbf{x}, W, V, W_2) = \big(\operatorname{Swish}_1(\mathbf{x}W) \otimes \mathbf{x}V\big) W_2",
                    "minimind": "对应 model/model_minimind.py 第 136-146 行的 FeedForward 类。"
                }
            ],
            "code_walkthrough": [
                ("model/model_minimind.py:L136-L146", "class FeedForward(nn.Module):\n    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):\n        super().__init__()\n        intermediate_size = intermediate_size or config.intermediate_size\n        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)\n        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)\n        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)\n        self.act_fn = ACT2FN[config.hidden_act]\n    def forward(self, x):\n        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))", "全层无偏置 (bias=False)，精简访存；允许传入自定义 intermediate_size 以便第 9 章 MoE 专家复用。")
            ],
            "hints": [
                "TODO(ch07-1) FeedForward.forward：return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))"
            ]
        },

        # ===================================================================== 7. 第 8 章
        {
            "id": "ch08",
            "stage": "阶段 2 · 模型架构",
            "badge": "核心必做",
            "difficulty": "Medium",
            "title": "第 8 章：组装 TinyMindForCausalLM、Weight Tying 与 64M 参数精算",
            "subtitle": "将 RMSNorm、Attention、RoPE、FFN 组装为与 MiniMind 100% 权重兼容的因果语言模型，掌握小模型 Scaling Law",
            "file": "model.py",
            "mm_path": "minimind/model/model_minimind.py#L183-L258",
            "visualizer": "param_calc",
            "todos": ["TODO(ch08-1)", "TODO(ch08-2)", "TODO(ch08-3)"],
            "theory_html": T(r"""
<h4>1. <code>MiniMind-3</code> (Dense 64M) 参数精确推导公式</h4>
<p>一个包含 $L$ 层、隐藏维度 $d$、词表 $V$、Query 头数 $H_q$、KV 头数 $H_{kv}$、头维度 $d_h = d/H_q$、FFN 中间维度 $I$、且开启输入输出词嵌入共享（<code>tie_word_embeddings=True</code>）的模型，其总参数量精确公式为：</p>
$$N_{\text{total}} = \underbrace{V d}_{\text{共享词嵌入}} + L \cdot \Big[ \underbrace{2 d (H_q d_h) + 2 d (H_{kv} d_h)}_{\text{GQA 四大投影矩阵}} + \underbrace{2 d_h}_{\text{QK-Norm}} + \underbrace{3 d I}_{\text{SwiGLU}} + \underbrace{2 d}_{\text{2个RMSNorm}} \Big] + \underbrace{d}_{\text{最终Norm}}$$
<p>代入 MiniMind-3 默认值（$V=6400, d=768, L=8, H_q=8, H_{kv}=4, d_h=96, I=2432$）：</p>
$$N_{\text{total}} = 4,915,200 + 8 \times (1,769,472 + 192 + 5,603,328 + 1,536) + 768 = \mathbf{63,912,192 \approx 63.91\text{M}}$$

<h4>2. 为什么小模型必须开启 Weight Tying（词嵌入与输出层权重共享）？</h4>
<p>在 <code>MiniMindForCausalLM.__init__</code> 中有这样一行关键代码：</p>
<pre><code>if self.config.tie_word_embeddings: self.model.embed_tokens.weight = self.lm_head.weight</code></pre>
<ul>
  <li><strong>数学与几何视角（Press & Wolf, 2017）</strong>：输入嵌入矩阵的第 $v$ 行 $\mathbf{e}_v$ 是词 $v$ 作为上下文输入时的向量表示；输出矩阵 <code>lm_head</code> 的第 $v$ 行 $\mathbf{w}_v$ 是预测下一个词是否为 $v$ 时的内积模板（$z_v = \mathbf{h}_t^T \mathbf{w}_v$）。在同一个语义度量空间中，$\mathbf{e}_v$ 与 $\mathbf{w}_v$ 理应重合。</li>
  <li><strong>梯度更新收益</strong>：如果不共享，生僻词作为输入出现时只更新 <code>embed_tokens</code>，作为预测目标出现时只更新 <code>lm_head</code>；共享后每一次出现都同时更新同一组向量，样本效率翻倍，同时直接省下 4.92M 参数！</li>
</ul>
"""),
            "papers": [
                {
                    "title": "Using the Output Embedding to Improve Language Models (Weight Tying)",
                    "authors": "Ofir Press, Lior Wolf",
                    "venue": "EACL 2017 · arXiv:1608.05859",
                    "pain": "输入词嵌入矩阵与输出 Softmax 投影矩阵尺寸完全相同 (V×d)，分开训练既浪费大量参数又导致低频词表征欠拟合。",
                    "formula": r"W_{\text{embed}} \equiv W_{\text{lm\_head}} \in \mathbb{R}^{V \times d}",
                    "minimind": "对应 model/model_minimind.py 第 30, 241, 247 行的 tie_word_embeddings=True。"
                },
                {
                    "title": "MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases",
                    "authors": "Zechun Liu et al. (Meta Reality Labs)",
                    "venue": "ICML 2024 · arXiv:2402.14905",
                    "pain": "Chinchilla 等经典 Scaling Law 面向百亿大模型，忽略了百兆级小模型中深度/宽度比例与权重共享的决定性作用。",
                    "formula": r"\text{Deep \& Thin } (L\uparrow, d\downarrow) + \text{Embedding Sharing} + \text{GQA} + \text{SwiGLU}",
                    "minimind": "MiniMind README 模型配置章节的核心参考论文，指导了 d_model=768, n_layers=8 的工程折中。"
                }
            ],
            "code_walkthrough": [
                ("model/model_minimind.py:L218-L224", "start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0\nhidden_states = self.dropout(self.embed_tokens(input_ids))\n...\nposition_embeddings = (self.freqs_cos[start_pos:start_pos + seq_length], self.freqs_sin[start_pos:start_pos + seq_length])", "当使用 KV Cache 逐词推理时，start_pos 等于已缓存的历史 token 长度，从预计算的 RoPE 表中精准切出当前位置对应的 [start_pos : start_pos + seq_length] 旋转角。"),
                ("model/model_minimind.py:L252-L253", "slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep\nlogits = self.lm_head(hidden_states[:, slice_indices, :])", "logits_to_keep 优化：在推理或 RL 采样时无需对前面所有 Prompt 位置乘以 lm_head，只切片最后需要的 token 计算词表投影，大幅节省峰值显存。")
            ],
            "hints": [
                "TODO(ch08-1) TinyMindBlock.forward：residual = hidden_states; hidden_states, present_key_value = self.self_attn(self.input_layernorm(hidden_states), position_embeddings, past_key_value, use_cache, attention_mask); hidden_states = hidden_states + residual; hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))",
                "TODO(ch08-2) RoPE 位置切片：position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_length], self.freqs_sin[start_pos:start_pos + seq_length])",
                "TODO(ch08-3) Shift 交叉熵损失：x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous(); loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)"
            ]
        },

        # ===================================================================== 8. 第 9 章
        {
            "id": "ch09",
            "stage": "阶段 2 · 模型架构",
            "badge": "进阶核心",
            "difficulty": "Hard",
            "title": "第 9 章：混合专家模型 MoE（Top-K 路由、STE 直通梯度与负载均衡损失）",
            "subtitle": "深入理解 198M-A64M 的稀疏激活机制、k=1 时的直通梯度估计器与 Switch 负载均衡辅助损失",
            "file": "moe.py",
            "mm_path": "minimind/model/model_minimind.py#L148-L181",
            "visualizer": "moe_vis",
            "todos": ["TODO(ch09-1)", "TODO(ch09-2)", "TODO(ch09-3)"],
            "theory_html": T(r"""
<h4>1. 稀疏激活（Sparse Activation）：如何做到参数量翻 3 倍而推理计算量不变？</h4>
<p>在 <code>MOEFeedForward</code> 中，我们将原本单一的 FFN 替换为 $E$ 个结构相同的专家网络 $\{\operatorname{Expert}_1, \dots, \operatorname{Expert}_E\}$（默认 $E=4$）和一个轻量级线性路由器 $W_{\text{gate}} \in \mathbb{R}^{d \times E}$。对于每个 token $\mathbf{x} \in \mathbb{R}^d$：</p>
$$\mathbf{p} = \operatorname{softmax}(W_{\text{gate}}^T \mathbf{x}) \in \mathbb{R}^E, \qquad \mathcal{T} = \operatorname{TopK}(\mathbf{p}, k)$$
$$\mathbf{y} = \sum_{i \in \mathcal{T}} \tilde{p}_i \cdot \operatorname{Expert}_i(\mathbf{x}), \qquad \tilde{p}_i = \frac{p_i}{\sum_{j \in \mathcal{T}} p_j}$$
<p>当 $E=4, k=1$ 时，每层 FFN 参数从 5.60M 扩充到 $4 \times 5.60\text{M} = 22.41\text{M}$（8 层总参数从 64M 增至 <strong>198M</strong>），但每个 token 只进入 1 个专家计算，单 token 激活参数仍为 <strong>64M</strong>（记为 <code>198M-A64M</code>）。</p>

<h4>2. 为什么当 $k=1$ 且开启权重归一化时，必须用 STE 直通梯度技巧？</h4>
<p>注意看上面的归一化公式 $\tilde{p}_i = p_i / \sum_{j \in \mathcal{T}} p_j$：当只选 $k=1$ 个专家 $\mathcal{T} = \{i^*\}$ 时，分子分母都是 $p_{i^*}$，约分后 $\tilde{p}_{i^*} \equiv 1.0$！</p>
<ul>
  <li>如果你在代码里直接写 <code>topk_weight = 1.0</code>，那么输出 $\mathbf{y} = 1.0 \cdot \operatorname{Expert}_{i^*}(\mathbf{x})$ 中完全不包含路由器概率 $p_{i^*}$，<strong>路由器 <code>self.gate</code> 的梯度将恒为 0，永远无法通过主任务交叉熵学习该把哪个 token 分给哪个专家</strong>！</li>
  <li>MiniMind 采用 <strong>Straight-Through Estimator (STE)</strong>：</li>
</ul>
$$\tilde{p}_{i^*} = p_{i^*} - \operatorname{sg}(p_{i^*}) + 1.0 \implies \begin{cases} \text{Forward: } \tilde{p}_{i^*} = 1.0 \\ \text{Backward: } \frac{\partial \mathbf{y}}{\partial p_{i^*}} = \operatorname{Expert}_{i^*}(\mathbf{x}) \end{cases}$$

<h4>3. Switch Transformer 负载均衡辅助损失（Load Balancing Aux Loss）推导</h4>
<p>为防止所有 token 都涌向同一个专家（专家坍塌），对本 batch 内全部 $T = B \times L$ 个 token 定义：</p>
<ul>
  <li><strong>专家 $i$ 的实际负载占比 $f_i$</strong>（不可微的离散计数）：$f_i = \frac{1}{T} \sum_{t=1}^T \mathbb{I}(i \in \mathcal{T}_t)$</li>
  <li><strong>专家 $i$ 的平均路由概率 $P_i$</strong>（可微）：$P_i = \frac{1}{T} \sum_{t=1}^T p_i(\mathbf{x}_t)$</li>
</ul>
$$\mathcal{L}_{\text{aux}} = \alpha \cdot E \sum_{i=1}^{E} f_i \cdot P_i$$
<p><strong>为什么最小化 $\sum f_i P_i$ 能实现均衡？</strong> 因为 $\sum_{i=1}^E f_i = 1$ 且 $\sum_{i=1}^E P_i = 1$，当某个专家负载过高（$f_i$ 大）时，梯度 $\frac{\partial \mathcal{L}_{\text{aux}}}{\partial P_i} = \alpha E f_i$ 就会赋予它最大的正惩罚，迫使路由器压低该专家的概率 $P_i$；当且仅当所有专家均匀分配 $f_i = P_i = 1/E$ 时，$E \sum f_i P_i$ 达到理论下界 $1$。</p>
"""),
            "papers": [
                {
                    "title": "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity",
                    "authors": "William Fedus, Barret Zoph, Noam Shazeer (Google Brain)",
                    "venue": "JMLR 2022 · arXiv:2101.03961",
                    "pain": "Shazeer (2017) 的传统 MoE 认为必须选 k >= 2 个专家做相对比较才能给路由器提供梯度，且专家负载极易失衡坍塌。",
                    "formula": r"\mathcal{L}_{\text{aux}} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i, \quad f_i = \frac{1}{T}\sum_{x\in\mathcal{B}} \mathbb{I}\{\arg\max p(x) = i\},\quad P_i = \frac{1}{T}\sum_{x\in\mathcal{B}} p_i(x)",
                    "minimind": "对应 model/model_minimind.py 第 148-181 行的 MOEFeedForward 类（默认 4 experts, top-1 routing, router_aux_loss_coef=5e-4）。"
                }
            ],
            "code_walkthrough": [
                ("model/model_minimind.py:L168-L175", "for i, expert in enumerate(self.experts):\n    mask = (topk_idx == i)\n    if mask.any():\n        token_idx = mask.any(dim=-1).nonzero().flatten()\n        weight = topk_weight[mask].view(-1, 1)\n        y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))\n    elif self.training:\n        y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())", "按专家分桶收集命中的 token_idx 批量送入 expert 计算，再用 index_add_ 累加回输出张量 y；未命中的专家通过 0 * sum(...) 挂入计算图以防 DDP 报错。"),
                ("model/model_minimind.py:L176-L178", "if self.training and self.config.router_aux_loss_coef > 0:\n    load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)\n    self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef", "严格对应 Switch Transformer 的辅助损失公式：load 为 f_i，scores.mean(0) 为 P_i。")
            ],
            "hints": [
                "TODO(ch09-1)：topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False)",
                "TODO(ch09-2)：token_idx = mask.any(dim=-1).nonzero().flatten(); weight = topk_weight[mask].view(-1, 1); y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))",
                "TODO(ch09-3)：load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0); self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef"
            ]
        },

        # ===================================================================== 9. 第 10 章
        {
            "id": "ch10",
            "stage": "阶段 2 · 模型架构",
            "badge": "核心必做",
            "difficulty": "Medium",
            "title": "第 10 章：自回归解码与五大采样策略（Temperature · 重复惩罚 · Top-K · Top-P）",
            "subtitle": "掌握核采样 (Nucleus Sampling) 的右移掩码细节与 CTRL 正负不对称重复惩罚公式",
            "file": "generate.py",
            "mm_path": "minimind/model/model_minimind.py#L261-L293",
            "visualizer": "sampling_vis",
            "todos": ["TODO(ch10-1)", "TODO(ch10-2)", "TODO(ch10-3)", "TODO(ch10-4)"],
            "theory_html": T(r"""
<h4>1. 为什么贪心搜索（Greedy Decoding）会导致神经文本退化（复读机现象）？</h4>
<p>Holtzman et al. (2020, ICLR) 研究发现，即使训练良好的语言模型，若每步都取 $\arg\max_v P(v \mid x_{<t})$，生成长文本时有极高概率陷入<strong>正反馈无限循环</strong>（一旦某句话重复了一次，自注意力机制会让下一次重复的条件概率更高）。因此现代 LLM 解码采用一套<strong>四级 Logits 变换流水线</strong>：</p>

<h4>2. 四级 Logits 滤波器数学公式</h4>
<ol>
  <li><strong>温度缩放（Temperature Scaling）</strong>：$z_i \leftarrow z_i / \tau$。从统计物理玻尔兹曼分布来看，$\tau \to 0^+$ 时分布坍缩至最大项（熵 $H \to 0$）；$\tau \to \infty$ 时趋向均匀分布（熵 $H \to \ln V$）。</li>
  <li><strong>CTRL 正负不对称重复惩罚（Keskar et al., 2019）</strong>：对已出现在上下文 $x_{<t}$ 中的词 $v \in \text{seen}$，设惩罚系数 $r \ge 1$：
    $$z_v \leftarrow \begin{cases} z_v / r & \text{if } z_v > 0 \\ z_v \cdot r & \text{if } z_v \le 0 \end{cases}$$
    为什么不能统一除以 $r$？因为当 $z_v < 0$（如 $-4.0$）时，除以 $r=2$ 会变成 $-2.0$（数值反而变大了，概率不降反升！）。分正负处理才能保证不论正负，变换后 $z_v$ 都单调减小。</li>
  <li><strong>Top-K 截断</strong>：令 $v_K$ 为第 $K$ 大的 logit 值，将所有 $z_i < v_K$ 置为 $-\infty$。</li>
  <li><strong>Top-P 核采样（Nucleus Sampling，Holtzman et al., 2020）</strong>：将词表按概率从大到小排序为 $p_{(1)} \ge p_{(2)} \ge \dots \ge p_{(V)}$，取累积概率首次达到或超过 $p$ 的下标 $k^* = \min\{k : \sum_{j=1}^k p_{(j)} > p\}$，保留前 $k^*$ 个词，把 $j > k^*$ 的位置全部置为 $-\infty$。
    <br>👉 <strong>代码关键细节</strong>：判断 <code>cumsum > top_p</code> 会把第 $k^*$ 个词自己也标为 <code>True</code>！因此必须用 <code>F.pad(mask[..., :-1], (1, 0))</code> 将掩码<strong>向右平移 1 位（首位补 False）</strong>，这样前 $k^*$ 个词均为 <code>False</code>（保留），从 $k^* + 1$ 开始才置为 $-\infty$！</li>
</ol>
"""),
            "papers": [
                {
                    "title": "The Curious Case of Neural Text Degeneration (Nucleus / Top-p Sampling)",
                    "authors": "Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, Yejin Choi",
                    "venue": "ICLR 2020 · arXiv:1904.09751",
                    "pain": "最大化似然解码（Beam Search / Greedy）生成的文本枯燥且极易陷入循环重复；而固定 K 的 Top-K 采样无法适应不同上下文下概率分布有时极尖锐、有时极平坦的变化。",
                    "formula": r"V^{(p)} = \arg\min_{V' \subset V} |V'| \quad \text{s.t.} \quad \sum_{x \in V'} P(x \mid x_{<t}) \ge p",
                    "minimind": "对应 model/model_minimind.py 第 279-282 行的 top_p 核采样实现。"
                },
                {
                    "title": "CTRL: A Conditional Transformer Language Model for Controllable Generation",
                    "authors": "Nitish Shirish Keskar, Bryan McCann, Lav R. Varshney, Caiming Xiong, Richard Socher",
                    "venue": "arXiv:1909.05858 (2019)",
                    "pain": "小模型或长文本生成时倾向于反复输出刚生成过的词。",
                    "formula": r"P_i = \frac{\exp(z_i / (T \cdot I(i \in g)))}{\sum_j \exp(z_j / (T \cdot I(j \in g)))}, \quad I(c) = \theta \text{ if } c \text{ else } 1",
                    "minimind": "对应 model/model_minimind.py 第 274-276 行的 repetition_penalty 实现。"
                }
            ],
            "code_walkthrough": [
                ("model/model_minimind.py:L270-L271", "past_len = past_key_values[0][0].shape[1] if past_key_values else 0\noutputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs)", "Prefill 阶段 past_len=0，喂入完整 Prompt；Decode 阶段 past_len=L-1，切片 input_ids[:, past_len:] 只喂入最后 1 个新生成的 token！"),
                ("model/model_minimind.py:L279-L282", "if do_sample and top_p < 1.0:\n    sorted_logits, sorted_indices = torch.sort(logits, descending=True)\n    mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p\n    logits[mask.scatter(1, sorted_indices, F.pad(mask[..., :-1], (1, 0)))] = -float('inf')", "F.pad(mask[..., :-1], (1, 0)) 把掩码右移一位确保第 1 个超过 top_p 的 token 也被保留，再用 scatter 按 sorted_indices 映射回原词表顺序。")
            ],
            "hints": [
                "TODO(ch10-1) apply_repetition_penalty：遍历 batch i，seen = torch.unique(input_ids[i]); score = logits[i, seen]; logits[i, seen] = torch.where(score > 0, score / penalty, score * penalty)",
                "TODO(ch10-2) top_k_filter：logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')",
                "TODO(ch10-3) top_p_filter：按 descending=True 排序，算 cumsum(softmax) > top_p，用 F.pad(mask[..., :-1], (1, 0)) 右移后 scatter 回原顺序置 -inf。",
                "TODO(ch10-4) KV Cache 增量切片：past_len = past_key_values[0][0].shape[1] if past_key_values else 0; outputs = model(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache)"
            ]
        },

        # ===================================================================== 10. 第 11 & 13 章
        {
            "id": "ch11-13",
            "stage": "阶段 3 · 预训练与微调",
            "badge": "核心必做",
            "difficulty": "Medium",
            "title": "第 11 & 13 章：Pretrain 与 SFT 数据掩码本质差异（Assistant-Only Loss Mask）",
            "subtitle": "彻底看清为什么 Pretrain 与 SFT 的模型和训练脚本一模一样，唯一区别在于 labels 中哪些位置被设为 -100",
            "file": "data.py",
            "mm_path": "minimind/dataset/lm_dataset.py#L40-L122",
            "visualizer": "mask_vis",
            "todos": ["TODO(ch11-1)", "TODO(ch13-1)"],
            "theory_html": T(r"""
<h4>1. Pretrain vs SFT 的数学目标对比</h4>
<p>给定一条多轮对话样本 $\mathbf{x} = (u_1, a_1, u_2, a_2)$，其中 $u_k$ 是用户提问（含 System 与模板头），$a_k$ 是 Assistant 的回复（含结束符 <code>&lt;|im_end|&gt;\n</code>）：</p>
<ul>
  <li><strong>无监督预训练（Pretrain）</strong>：最大化整段文本的联合概率（只排除末尾 Padding）：
    $$\mathcal{L}_{\text{Pretrain}} = -\sum_{t \in \text{non-pad}} \log P_\theta(x_t \mid x_{<t})$$</li>
  <li><strong>有监督指令微调（SFT）</strong>：只最大化给定上下文时 <strong>Assistant 回复区间 $\mathcal{A} = \bigcup_k a_k$</strong> 的条件概率：
    $$\mathcal{L}_{\text{SFT}} = -\sum_{t \in \mathcal{A}} \log P_\theta(x_t \mid x_{<t})$$</li>
</ul>
<p>在 PyTorch 实现中，不需要修改模型或损失函数的一行代码——只需要在构造 <code>labels</code> 时，把所有不属于 $\mathcal{A}$ 的位置全部填为 <strong><code>-100</code></strong>！</p>

<h4>2. 子序列滑动窗口匹配算法（<code>generate_labels</code>）</h4>
<p>MiniMind 的 <code>SFTDataset</code> 先算出两个边界标记的 token ID 序列：</p>
<ul>
  <li>起始锚点 <code>bos_id</code> = <code>tokenizer("&lt;|im_start|&gt;assistant\n").input_ids</code>（注意：锚点本身属于提示头，<strong>不计入 Loss</strong>）</li>
  <li>结束锚点 <code>eos_id</code> = <code>tokenizer("&lt;|im_end|&gt;\n").input_ids</code>（注意：结束锚点<strong>必须计入 Loss</strong>，否则模型永远学不会在回答完后停止！）</li>
</ul>
<p>用指针 $i$ 扫描 <code>input_ids</code>：一旦匹配到 <code>input_ids[i : i+len(bos_id)] == bos_id</code>，令 $\text{start} = i + \text{len}(\text{bos\_id})$，向后搜索 <code>eos_id</code> 的起始位置 $\text{end}$，然后将闭开区间 $[\text{start},\; \min(\text{end} + \text{len}(\text{eos\_id}),\, L_{\max}))$ 内的 <code>labels[j] = input_ids[j]</code>！</p>
"""),
            "papers": [
                {
                    "title": "Training language models to follow instructions with human feedback (InstructGPT)",
                    "authors": "Long Ouyang et al. (OpenAI)",
                    "venue": "NeurIPS 2022 · arXiv:2203.02155",
                    "pain": "纯粹用 Next-Token Prediction 预训练的 GPT-3 基座模型在收到用户问题时，往往接着续写更多的问题，而不是作为助手回答问题。",
                    "formula": r"\mathcal{L}_{\text{SFT}}(\theta) = -\mathbb{E}_{(x, y) \sim \mathcal{D}_{\text{SFT}}} \left[ \sum_{t=1}^{|y|} \log \pi_\theta(y_t \mid x, y_{<t}) \right]",
                    "minimind": "对应 dataset/lm_dataset.py 第 61-122 行的 SFTDataset 与 trainer/train_full_sft.py。"
                }
            ],
            "code_walkthrough": [
                ("dataset/lm_dataset.py:L91-L107", "def generate_labels(self, input_ids):\n    labels = [-100] * len(input_ids)\n    i = 0\n    while i < len(input_ids):\n        if input_ids[i:i + len(self.bos_id)] == self.bos_id:\n            start = i + len(self.bos_id)\n            end = start\n            while end < len(input_ids):\n                if input_ids[end:end + len(self.eos_id)] == self.eos_id:\n                    break\n                end += 1\n            for j in range(start, min(end + len(self.eos_id), self.max_length)):\n                labels[j] = input_ids[j]\n            i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)\n        else:\n            i += 1\n    return labels", "支持任意多轮对话：一段样本里有几轮 assistant 回复，就会激活几段连续的 label，单次前向传播同时训练所有轮次。")
            ],
            "hints": [
                "TODO(ch11-1) PretrainDataset.__getitem__：分词截断至 max_length-2 -> 前后拼 [bos] + tokens + [eos] -> 右侧补 pad 到 max_length -> 转 tensor -> labels = input_ids.clone() 并将 labels[input_ids == pad_token_id] = -100。",
                "TODO(ch13-1) generate_sft_labels：参考上方 generate_labels 源码，注意当传入 keep_value 不为 None 时（供 DPO 复用生成 0/1 mask），写入 keep_value 否则写入 input_ids[j]。"
            ]
        },

        # ===================================================================== 11. 第 12 章
        {
            "id": "ch12",
            "stage": "阶段 3 · 预训练与微调",
            "badge": "核心必做",
            "difficulty": "Medium",
            "title": "第 12 章：现代 LLM 训练循环工程（余弦退火 · 梯度累积 · 梯度裁剪）",
            "subtitle": "从数学上证明为什么梯度累积必须将单步 Loss 除以 K，掌握全局 L2 梯度范数裁剪与余弦学习率调度",
            "file": "train_utils.py",
            "mm_path": "minimind/trainer/train_pretrain.py#L24-L72",
            "visualizer": "fp_roofline",
            "todos": ["TODO(ch12-1)", "TODO(ch12-2)", "TODO(ch12-3)"],
            "theory_html": T(r"""
<h4>1. 余弦退火学习率调度（Cosine Annealing Schedule）</h4>
<p>MiniMind 在整个训练周期 $T = \text{epochs} \times \text{iters}$ 内按 step 更新学习率，从初始值 $\eta_0$ 平滑衰减到 $0.1 \eta_0$：</p>
$$\eta(t) = \eta_0 \cdot \left( 0.1 + 0.45 \cdot \left(1 + \cos\frac{\pi t}{T}\right) \right)$$
<p>代入边界验证：当 $t=0$ 时 $\cos 0 = 1 \implies \eta(0) = \eta_0(0.1 + 0.9) = \eta_0$；当 $t=T$ 时 $\cos \pi = -1 \implies \eta(T) = 0.1\eta_0$。</p>

<h4>2. 梯度累积（Gradient Accumulation）的严格等价性证明</h4>
<p>假设显存只能放下微批大小（Micro-Batch Size）$b = 32$，但我们希望等价于大批大小 $B = K \cdot b = 256$（累积步数 $K = 8$）。大 Batch 的平均损失与梯度为：</p>
$$\mathcal{L}_{\text{big}} = \frac{1}{K b} \sum_{k=1}^{K} \sum_{i=1}^{b} \ell_{k, i} = \frac{1}{K} \sum_{k=1}^{K} \underbrace{\left(\frac{1}{b} \sum_{i=1}^{b} \ell_{k, i}\right)}_{\mathcal{L}_k} \implies \nabla_\theta \mathcal{L}_{\text{big}} = \sum_{k=1}^{K} \nabla_\theta \left(\frac{\mathcal{L}_k}{K}\right)$$
<p>在 PyTorch 中，只要不调用 <code>optimizer.zero_grad()</code>，每次调用 <code>.backward()</code> 都会把新梯度<strong>累加（<code>+=</code>）</strong>到 <code>param.grad</code> 上。因此，<strong>每一步把微批损失 $\mathcal{L}_k$ 先除以 $K$ 再调 <code>backward()</code>，连续累加 $K$ 步后的 <code>param.grad</code> 在数学上与一次性跑 $B = K b$ 大 Batch 的梯度 100% 相等</strong>！</p>

<h4>3. 全局梯度范数裁剪（Global Gradient Norm Clipping）</h4>
<p>令模型所有参数的梯度拼接为一个大向量 $\mathbf{g}$，其全局 $L_2$ 范数为 $\|\mathbf{g}\|_2 = \sqrt{\sum_p \|\nabla_p \mathcal{L}\|_2^2}$。给定阈值 $c = 1.0$：</p>
$$\mathbf{g} \leftarrow \mathbf{g} \cdot \min\!\left(1,\; \frac{c}{\|\mathbf{g}\|_2 + 10^{-6}}\right)$$
<p>注意它<strong>只缩放梯度的长度而不改变梯度的方向</strong>，有效防止个别脏样本产生的梯度尖峰破坏已收敛的权重。</p>
"""),
            "papers": [
                {
                    "title": "SGDR: Stochastic Gradient Descent with Warm Restarts (Cosine Annealing)",
                    "authors": "Ilya Loshchilov, Frank Hutter",
                    "venue": "ICLR 2017 · arXiv:1608.03983",
                    "pain": "阶梯式学习率衰减（Step Decay）在跳变点会引起训练震荡，且需要针对不同步数手工调参。",
                    "formula": r"\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{T_{\text{cur}}}{T_{\max}}\pi\right)\right)",
                    "minimind": "对应 trainer/trainer_utils.py 第 42-43 行的 get_lr 函数（取 η_min = 0.1*lr, η_max = lr）。"
                }
            ],
            "code_walkthrough": [
                ("trainer/train_pretrain.py:L35-L47", "loss = res.loss + res.aux_loss\nloss = loss / args.accumulation_steps\nscaler.scale(loss).backward()\nif step % args.accumulation_steps == 0 or step == iters:\n    scaler.unscale_(optimizer)\n    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)\n    scaler.step(optimizer)\n    scaler.update()\n    optimizer.zero_grad(set_to_none=True)", "注意顺序：① loss 除以 accumulation_steps -> ② scale.backward() -> ③ 满 K 步后先 unscale_ 把梯度还原为真实量级 -> ④ 在真实量级上做 clip_grad_norm_ -> ⑤ step & update & zero_grad。")
            ],
            "hints": [
                "TODO(ch12-1) get_lr：return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))",
                "TODO(ch12-2)：loss = loss / accumulation_steps",
                "TODO(ch12-3)：if step % accumulation_steps == 0 or step == iters: 若有 scaler 先 scaler.unscale_(optimizer)，然后 torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)，再 step() 并 optimizer.zero_grad(set_to_none=True)。"
            ]
        },

        # ===================================================================== 12. 第 14 章
        {
            "id": "ch14",
            "stage": "阶段 4 · 微调与蒸馏",
            "badge": "核心必做",
            "difficulty": "Medium",
            "title": "第 14 章：70 行从零手写 LoRA（低秩分解 · 非对称初始化 · 闭包绑定 · 权重合并）",
            "subtitle": "不依赖 peft 库，从内在秩假说与鞍点梯度分析出发，亲手实现 LoRA 动态挂载与无损合并",
            "file": "lora.py",
            "mm_path": "minimind/model/model_lora.py#L6-L70",
            "visualizer": "lora_kd",
            "todos": ["TODO(ch14-1)", "TODO(ch14-2)", "TODO(ch14-3)"],
            "theory_html": T(r"""
<h4>1. 内在秩假说（Intrinsic Rank Hypothesis）与 LoRA 矩阵分解</h4>
<p>对于预训练线性层权重 $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$，全参数微调学习增量 $\Delta W$ 需要更新 $d_{\text{out}} \times d_{\text{in}}$ 个参数。Hu et al. (2021) 提出将增量约束为秩 $r \ll \min(d_{\text{in}}, d_{\text{out}})$ 的两个低秩矩阵乘积：</p>
$$\mathbf{h} = W_0 \mathbf{x} + \Delta W \mathbf{x} = W_0 \mathbf{x} + \frac{\alpha}{r} B A \mathbf{x}, \qquad A \in \mathbb{R}^{r \times d_{\text{in}}},\; B \in \mathbb{R}^{d_{\text{out}} \times r}$$

<h4>2. 为什么必须「$A$ 高斯初始化、$B$ 全零初始化」？（鞍点梯度严格证明）</h4>
<p>设损失函数对输出 $\mathbf{h}$ 的上游梯度为 $\mathbf{g} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}} \in \mathbb{R}^{d_{\text{out}}}$。根据矩阵求导链式法则，$A$ 和 $B$ 的梯度分别为：</p>
$$\frac{\partial \mathcal{L}}{\partial B} = \frac{\alpha}{r} \mathbf{g} (A\mathbf{x})^T, \qquad \frac{\partial \mathcal{L}}{\partial A} = \frac{\alpha}{r} (B^T \mathbf{g}) \mathbf{x}^T$$
<ul>
  <li><strong>若 $A = \mathbf{0}$ 且 $B = \mathbf{0}$</strong>：代入上式得 $\frac{\partial \mathcal{L}}{\partial B} = \mathbf{0}$ 且 $\frac{\partial \mathcal{L}}{\partial A} = \mathbf{0}$！参数永远卡死在零梯度鞍点，一步都动不了！</li>
  <li><strong>若 $A \sim \mathcal{N}(0, \sigma^2)$ 且 $B \sim \mathcal{N}(0, \sigma^2)$</strong>：训练第 0 步时 $\Delta W = \frac{\alpha}{r}BA \ne \mathbf{0}$，相当于往已收敛的预训练模型里直接掺入随机噪声，破坏初始表征。</li>
  <li><strong>非对称初始化（$A \sim \mathcal{N}(0, 0.02^2),\; B = \mathbf{0}$）</strong>：
    <br>① 第 0 步前向时：$\Delta W = \frac{\alpha}{r} \mathbf{0} \cdot A = \mathbf{0}$，<strong>模型初始输出与原模型 100% 完全一致</strong>；
    <br>② 第 1 步反向时：因为 $A\mathbf{x} \ne \mathbf{0}$，<strong>$\frac{\partial \mathcal{L}}{\partial B} = \frac{\alpha}{r}\mathbf{g}(A\mathbf{x})^T \ne \mathbf{0}$</strong>，$B$ 在第一步就立刻获得非零更新；从第 2 步起 $B \ne \mathbf{0}$，$A$ 也随之开始更新！</li>
</ul>

<h4>3. 推理零延迟：线性叠加合并</h4>
<p>训练完毕后，利用矩阵乘法对加法的分配律，直接将低秩旁路永久合入主干权重：</p>
$$W_{\text{merged}} = W_0 + \frac{\alpha}{r} B A \implies W_{\text{merged}}\mathbf{x} \equiv W_0\mathbf{x} + \frac{\alpha}{r}BA\mathbf{x}$$
"""),
            "papers": [
                {
                    "title": "LoRA: Low-Rank Adaptation of Large Language Models",
                    "authors": "Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (Microsoft)",
                    "venue": "ICLR 2022 · arXiv:2106.09685",
                    "pain": "全量微调大模型需要复制整个模型权重并维护庞大的 Adam 优化器状态；而 Adapter 层又会增加串行深度导致推理延迟上升。",
                    "formula": r"h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} B A x, \quad A \sim \mathcal{N}(0, \sigma^2),\; B = 0",
                    "minimind": "对应 model/model_lora.py 全部 70 行纯手写实现与 trainer/train_lora.py。"
                }
            ],
            "code_walkthrough": [
                ("model/model_lora.py:L28-L34", "original_forward = module.forward\n# 显式绑定\ndef forward_with_lora(x, layer1=original_forward, layer2=lora):\n    return layer1(x) + layer2(x)\nmodule.forward = forward_with_lora", "用默认参数 layer1=original_forward, layer2=lora 在函数定义那一刻绑定当前层的对象，避开 Python for 循环闭包的晚绑定陷阱。"),
                ("model/model_lora.py:L66-L68", "if hasattr(module, 'lora'):\n    delta = (module.lora.B.weight.data.float() @ module.lora.A.weight.data.float()) * module.lora.scaling.float()\n    state_dict[f'{name}.weight'] += delta.cpu()", "注意 nn.Linear(in, out) 的 weight 形状是 [out, in]，所以 B.weight [out, r] @ A.weight [r, in] 恰好等于 [out, in]，直接加到 module.weight 上！")
            ],
            "hints": [
                "TODO(ch14-1)：self.A.weight.data.normal_(mean=0.0, std=0.02); self.B.weight.data.zero_()",
                "TODO(ch14-2)：return self.B(self.A(x)) * self.scaling",
                "TODO(ch14-3) merge_lora：delta = (module.lora.B.weight.float() @ module.lora.A.weight.float()) * module.lora.scaling.float(); state_dict[f'{name}.weight'] = (module.weight.float() + delta).to(module.weight.dtype)"
            ]
        },

        # ===================================================================== 13. 第 15 章
        {
            "id": "ch15",
            "stage": "阶段 4 · 微调与蒸馏",
            "badge": "核心必做",
            "difficulty": "Medium",
            "title": "第 15 章：白盒知识蒸馏（暗知识 Soft Targets 与 T² 梯度量级补偿推导）",
            "subtitle": "从一阶泰勒展开严格证明为什么高温软化 KL 散度会导致梯度缩小 1/T²，实现师生双模型联合蒸馏损失",
            "file": "distill.py",
            "mm_path": "minimind/trainer/train_distillation.py#L24-L98",
            "visualizer": "lora_kd",
            "todos": ["TODO(ch15-1)"],
            "theory_html": T(r"""
<h4>1. 暗知识（Dark Knowledge）与温度软化分布</h4>
<p>在硬标签 SFT 中，目标是一个 One-Hot 分布（正确词概率为 1，其余 6399 个词概率全为 0）。但正如 Hinton et al. (2015) 指出的，教师模型在其余 6399 个词上赋予的<strong>相对概率大小</strong>蕴含着极其丰富的语义结构（例如在句子 <code>"清晨的阳光洒在___"</code> 后，教师给 <code>"窗台"</code> 0.6、给 <code>"桌面"</code> 0.25、给 <code>"草地"</code> 0.14、而给 <code>"钢筋"</code> $10^{-8}$）。引入温度 $T \ge 1$ 软化师生双方的输出分布：</p>
$$p_i^{(t)}(T) = \frac{\exp(z_i^{(t)} / T)}{\sum_j \exp(z_j^{(t)} / T)}, \qquad p_i^{(s)}(T) = \frac{\exp(z_i^{(s)} / T)}{\sum_j \exp(z_j^{(s)} / T)}$$

<h4>2. 为什么 KL 散度项前面必须乘以 $T^2$？（泰勒展开严格证明）</h4>
<p>记软标签 KL 散度为 $\mathcal{L}_{\text{KL}}(T) = \mathrm{KL}\big(\mathbf{p}^{(t)}(T) \parallel \mathbf{p}^{(s)}(T)\big) = \sum_{j=1}^V p_j^{(t)}(T) \log \frac{p_j^{(t)}(T)}{p_j^{(s)}(T)}$。对学生模型的第 $i$ 个 logit $z_i^{(s)}$ 求偏导（注意系数 $1/T$ 来自复合函数求导）：</p>
$$\frac{\partial \mathcal{L}_{\text{KL}}(T)}{\partial z_i^{(s)}} = \frac{1}{T} \left( p_i^{(s)}(T) - p_i^{(t)}(T) \right)$$
<p>当温度 $T$ 较大时，对指数项做一阶泰勒近似 $e^{z_i / T} \approx 1 + z_i / T$，且假设 logits 零均值（$\sum_j z_j = 0$），则：</p>
$$p_i(T) \approx \frac{1 + z_i / T}{V} \implies p_i^{(s)}(T) - p_i^{(t)}(T) \approx \frac{1}{V T} \left(z_i^{(s)} - z_i^{(t)}\right)$$
<p>代回梯度公式：</p>
$$\frac{\partial \mathcal{L}_{\text{KL}}(T)}{\partial z_i^{(s)}} \approx \frac{1}{V \mathbf{T^2}} \left(z_i^{(s)} - z_i^{(t)}\right)$$
<p><strong>结论一目了然</strong>：温度 $T$ 会让 KL 散度对学生 logits 的梯度以 <strong>$1/T^2$</strong> 的比例缩小！为了在与硬标签交叉熵 $\mathcal{L}_{\text{CE}}$（其梯度是 $\mathcal{O}(1)$ 量级）加权求和时保持梯度量级不随超参数 $T$ 剧烈变化，<strong>必须在 $\mathcal{L}_{\text{KL}}(T)$ 前乘上补偿因子 $T^2$</strong>：</p>
$$\mathcal{L}_{\text{total}} = \alpha \mathcal{L}_{\text{CE}} + (1 - \alpha) T^2 \mathrm{KL}\big(\mathbf{p}^{(t)}(T) \parallel \mathbf{p}^{(s)}(T)\big)$$
"""),
            "papers": [
                {
                    "title": "Distilling the Knowledge in a Neural Network",
                    "authors": "Geoffrey Hinton, Oriol Vinyals, Jeff Dean (Google)",
                    "venue": "NIPS 2014 Deep Learning Workshop / arXiv:1503.02531 (2015)",
                    "pain": "庞大的集成模型或深层大模型推理成本高，而直接用硬标签在小模型上从头训练无法继承大模型学到的类别间相似性泛化结构。",
                    "formula": r"\frac{\partial C}{\partial z_i} = \frac{1}{T}(q_i - p_i) \approx \frac{1}{N T^2}(z_i - v_i) \implies \mathcal{L}_{\text{KD}} = T^2 \mathrm{KL}(p^T \parallel q^T)",
                    "minimind": "对应 trainer/train_distillation.py 第 24-38 行的 distillation_loss 函数。"
                }
            ],
            "code_walkthrough": [
                ("trainer/train_distillation.py:L24-L38", "def distillation_loss(student_logits, teacher_logits, temperature=1.0, reduction='batchmean'):\n    with torch.no_grad():\n        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()\n    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)\n    kl = F.kl_div(student_log_probs, teacher_probs, reduction=reduction)\n    return (temperature ** 2) * kl", "注意 PyTorch 的 F.kl_div(input, target) 约定第一个参数 input 是对数概率 log_softmax (学生)，第二个参数 target 是普通概率 softmax (老师)！")
            ],
            "hints": [
                "TODO(ch15-1) distillation_loss：在 torch.no_grad() 下算 teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()，然后 student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)，返回 (temperature ** 2) * F.kl_div(student_log_probs, teacher_probs, reduction=reduction)。"
            ]
        },

        # ===================================================================== 14. 第 16 章
        {
            "id": "ch16",
            "stage": "阶段 5 · 强化学习对齐",
            "badge": "核心必做",
            "difficulty": "Medium",
            "title": "第 16 章：策略优化统一视角与 DPO 直接偏好优化",
            "subtitle": "从带 KL 约束的强化学习目标出发，完整推导 DPO 如何消去配分函数 Z(x) 把 RL 变成解析二元交叉熵",
            "file": "dpo.py",
            "mm_path": "minimind/trainer/train_dpo.py#L25-L84",
            "visualizer": "rl_po",
            "todos": ["TODO(ch16-1)", "TODO(ch16-2)"],
            "theory_html": T(r"""
<h4>1. 从 RLHF 目标到最优策略的解析解（DPO 完整三步推导）</h4>
<p><strong>第一步（KL 约束下的奖励最大化目标）</strong>：在经典 RLHF 中，给定奖励函数 $r(x, y)$ 和参考模型 $\pi_{\text{ref}}$（通常为 SFT 模型），我们要寻找最大化期望奖励且不偏离 $\pi_{\text{ref}}$ 太远的策略 $\pi$：</p>
$$\max_{\pi} \; \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi(\cdot \mid x)} \big[ r(x, y) \big] - \beta \, \mathbb{D}_{\mathrm{KL}}\big( \pi(y \mid x) \parallel \pi_{\text{ref}}(y \mid x) \big)$$
<p>展开 KL 散度并提取系数 $-\beta$，目标等价于最小化某个吉布斯分布与 $\pi$ 的 KL 散度，由此直接写出<strong>全局唯一解析最优策略 $\pi^*(y \mid x)$</strong>：</p>
$$\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\!\left(\frac{1}{\beta} r(x, y)\right), \qquad Z(x) = \sum_{y} \pi_{\text{ref}}(y \mid x) \exp\!\left(\frac{1}{\beta} r(x, y)\right)$$
<p><strong>第二步（用最优策略反解奖励函数）</strong>：对上式两边取对数，把 $r(x, y)$ 解出来：</p>
$$r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$
<p><strong>第三步（代入 Bradley-Terry 偏好模型，消去棘手的配分函数 $Z(x)$！）</strong>：人类对同一问题 $x$ 的两个回答（胜者 $y_w$ vs 败者 $y_l$）的偏好概率服从 Bradley-Terry 模型 $P(y_w \succ y_l \mid x) = \sigma\big(r(x, y_w) - r(x, y_l)\big)$。将第二步的 $r(x, y)$ 代入差值，<strong>只与 $x$ 有关的 $\beta \log Z(x)$ 完美相减抵消</strong>！直接得到 DPO 损失函数：</p>
$$\mathcal{L}_{\mathrm{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \left( \underbrace{\log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)}}_{\text{Chosen 隐式奖励 }\hat{r}_w} - \underbrace{\log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}}_{\text{Rejected 隐式奖励 }\hat{r}_l} \right) \right) \right]$$

<h4>2. DPO 的梯度直觉：为什么说「你的语言模型本身就是一个奖励模型」？</h4>
<p>对 $\mathcal{L}_{\mathrm{DPO}}$ 求关于 $\theta$ 的梯度：</p>
$$\nabla_\theta \mathcal{L}_{\mathrm{DPO}} = -\beta \, \mathbb{E} \Big[ \underbrace{\sigma(\hat{r}_l - \hat{r}_w)}_{\text{当前模型判错的程度（权重）}} \cdot \big( \underbrace{\nabla_\theta \log \pi_\theta(y_w \mid x)}_{\text{提高好回答概率}} - \underbrace{\nabla_\theta \log \pi_\theta(y_l \mid x)}_{\text{压低坏回答概率}} \big) \Big]$$
<p>当模型给坏回答的隐式奖励高于好回答（$\hat{r}_l > \hat{r}_w$）时，权重 $\sigma(\hat{r}_l - \hat{r}_w)$ 自动放大，驱动模型强力拉开正负样本的对数概率差！</p>
"""),
            "papers": [
                {
                    "title": "Direct Preference Optimization: Your Language Model is Secretly a Reward Model (DPO)",
                    "authors": "Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn (Stanford)",
                    "venue": "NeurIPS 2023 · arXiv:2305.18290",
                    "pain": "RLHF-PPO 流程极其沉重：需先训练独立的 Reward Model，再在内存中同时加载 4 个大模型做在线采样与 Critic 拟合，显存开销大且极易训崩。",
                    "formula": r"\mathcal{L}_{\mathrm{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l)\sim\mathcal{D}}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta \log \frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right)\right]",
                    "minimind": "对应 trainer/train_dpo.py 第 25-51 行的 logits_to_log_probs 与 dpo_loss。"
                }
            ],
            "code_walkthrough": [
                ("trainer/train_dpo.py:L34-L50", "def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):\n    ref_log_probs = (ref_log_probs * mask).sum(dim=1)\n    policy_log_probs = (policy_log_probs * mask).sum(dim=1)\n    batch_size = ref_log_probs.shape[0]\n    chosen_ref_log_probs = ref_log_probs[:batch_size // 2]\n    reject_ref_log_probs = ref_log_probs[batch_size // 2:]\n    chosen_policy_log_probs = policy_log_probs[:batch_size // 2]\n    reject_policy_log_probs = policy_log_probs[batch_size // 2:]\n    pi_logratios = chosen_policy_log_probs - reject_policy_log_probs\n    ref_logratios = chosen_ref_log_probs - reject_ref_log_probs\n    logits = pi_logratios - ref_logratios\n    loss = -F.logsigmoid(beta * logits)\n    return loss.mean()", "注意代数恒等变换：(chosen_pi - reject_pi) - (chosen_ref - reject_ref) 恒等于 (chosen_pi - chosen_ref) - (reject_pi - reject_ref)，且 F.logsigmoid 在大负数下具备数值稳定性。")
            ],
            "hints": [
                "TODO(ch16-1) logits_to_log_probs：log_probs = F.log_softmax(logits, dim=2); return torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)",
                "TODO(ch16-2) dpo_loss：完全按照上方源码解析实现即可（先乘以 mask 在 dim=1 求和，再对半切分 chosen/rejected，算 -F.logsigmoid(beta * logits).mean()）。"
            ]
        },

        # ===================================================================== 15. 第 17 章
        {
            "id": "ch17",
            "stage": "阶段 5 · 强化学习对齐",
            "badge": "进阶核心",
            "difficulty": "Hard",
            "title": "第 17 章：PPO 近端策略优化与 GAE 广义优势估计",
            "subtitle": "吃透 Actor-Critic 四模型架构、末尾稀疏奖励向每个 token 分配的 GAE 倒序递推与 Schulman k3 KL 估计器",
            "file": "ppo.py",
            "mm_path": "minimind/trainer/train_ppo.py#L36-L245",
            "visualizer": "rl_po",
            "todos": ["TODO(ch17-1)", "TODO(ch17-2)", "TODO(ch17-3)"],
            "theory_html": T(r"""
<h4>1. 为什么在序列末尾只有一个奖励分，却能给前面每一个 token 算优势？（GAE 推导）</h4>
<p>在语言模型 RL 中，把每个生成步 $t \in \{0, \dots, T-1\}$ 看作马尔可夫决策过程（MDP）：状态 $s_t = (q, o_{<t})$，动作 $a_t = o_t$。外部奖励模型（Reward Model）只在整条回答结束（$t = T-1$）时给出一个标量总分 $R$，前面所有步的即时奖励均为 0：</p>
$$r_t = \begin{cases} 0 & 0 \le t < T - 1 \\ R & t = T - 1 \end{cases}$$
<p>Critic 价值网络 $V_\phi(s_t)$ 预测从位置 $t$ 往后写完的期望总得分。定义第 $t$ 步的 <strong>TD 残差（Temporal-Difference Error）</strong> $\delta_t$（其中约定终止后 $V(s_T) = 0$）：</p>
$$\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$
<ul>
  <li>若只用单步 $\hat{A}_t^{(1)} = \delta_t$：方差极小，但高度依赖 Critic 的准确性（<strong>高偏差 High Bias</strong>）。</li>
  <li>若直接用蒙特卡洛全回报 $\hat{A}_t^{(\infty)} = \sum_{k=0}^{T-1-t} \gamma^k \delta_{t+k} = \gamma^{T-1-t} R - V_\phi(s_t)$：无偏，但把整条句子的成败不加区分地归因给每一个词（<strong>高方差 High Variance</strong>）。</li>
  <li><strong>GAE（Schulman et al., 2016）</strong>用指数衰减权重 $\lambda \in [0, 1]$（默认 $\gamma=1, \lambda=0.95$）对所有 $k$-步估计做加权平均，化简为极其优雅的<strong>从后往前一步递推公式</strong>：</li>
</ul>
$$\hat{A}_t^{\mathrm{GAE}(\gamma, \lambda)} = \sum_{l=0}^{T-1-t} (\gamma \lambda)^l \delta_{t+l} = \delta_t + \gamma \lambda \hat{A}_{t+1}^{\mathrm{GAE}}, \qquad R_t^{\text{target}} = \hat{A}_t^{\mathrm{GAE}} + V_{\text{old}}(s_t)$$

<h4>2. 重要性采样与 PPO-Clip 截断代理目标</h4>
<p>采样一批回答后，PPO 会在同一批数据上做多轮小批量更新（<code>ppo_update_iters</code>）。为用旧策略 $\pi_{\text{old}}$ 采出的数据估计新策略 $\pi_\theta$ 的梯度，引入重要性采样概率比 $r_t(\theta) = \frac{\pi_\theta(o_t \mid q, o_{<t})}{\pi_{\text{old}}(o_t \mid q, o_{<t})} = \exp(\log\pi_\theta - \log\pi_{\text{old}})$，并在超出信任域 $[1-\epsilon, 1+\epsilon]$ 时截断梯度：</p>
$$\mathcal{L}_t^{\mathrm{CLIP}}(\theta) = -\min\!\Big( r_t(\theta) \hat{A}_t,\; \operatorname{clip}\big(r_t(\theta),\, 1-\epsilon,\, 1+\epsilon\big) \hat{A}_t \Big) = \max\!\Big( -r_t(\theta) \hat{A}_t,\; -\operatorname{clip}(r_t, 1-\epsilon, 1+\epsilon)\hat{A}_t \Big)$$

<h4>3. Schulman $k_3$ 无偏正定 KL 散度估计器</h4>
<p>为什么 MiniMind 在 PPO 和 GRPO 中都用 $e^{\Delta} - \Delta - 1$（其中 $\Delta = \log\pi_{\text{ref}}(o_t) - \log\pi_\theta(o_t)$）来计算 KL 惩罚，而不是直接用 $-\Delta = \log\pi_\theta - \log\pi_{\text{ref}}$？</p>
$$\widehat{\mathrm{KL}}_{k_3}(o_t) = e^{\Delta} - \Delta - 1 = \frac{\pi_{\text{ref}}(o_t)}{\pi_\theta(o_t)} - \log\frac{\pi_{\text{ref}}(o_t)}{\pi_\theta(o_t)} - 1 \ge 0$$
<p>因为：<strong>① 恒非负</strong>（由凸函数不等式 $e^x \ge 1 + x$，每一项都 $\ge 0$，绝不会在单样本上出现荒谬的负 KL 值）；<strong>② 严格无偏</strong>（在 $o_t \sim \pi_\theta$ 下求期望：$\mathbb{E}_{\pi_\theta}[\pi_{\text{ref}}/\pi_\theta - 1] = \sum \pi_{\text{ref}} - 1 = 0$，期望严格等于 $\mathbb{E}_{\pi_\theta}[-\Delta] = \mathrm{KL}(\pi_\theta \parallel \pi_{\text{ref}})$）；<strong>③ 低方差</strong>（在 $\Delta \approx 0$ 附近等于 $\frac{1}{2}\Delta^2$）。</p>
"""),
            "papers": [
                {
                    "title": "High-Dimensional Continuous Control Using Generalized Advantage Estimation (GAE)",
                    "authors": "John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, Pieter Abbeel (UC Berkeley)",
                    "venue": "ICLR 2016 · arXiv:1506.02438",
                    "pain": "策略梯度强化学习中，稀疏奖励归因给长序列每一步动作时方差极大，导致训练需要海量样本且极不稳定。",
                    "formula": r"\hat{A}_t^{\mathrm{GAE}(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}^V, \quad \delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)",
                    "minimind": "对应 trainer/train_ppo.py 第 144-151 行的 reversed(range(gen_len)) GAE 递推循环。"
                },
                {
                    "title": "Proximal Policy Optimization Algorithms (PPO)",
                    "authors": "John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov (OpenAI)",
                    "venue": "arXiv:1707.06347 (2017)",
                    "pain": "标准策略梯度在同一批采样数据上做多步更新会引起策略剧烈跳变崩溃；而 TRPO 需要计算二阶 Fisher 信息矩阵共轭梯度，无法扩展到大模型。",
                    "formula": r"L^{\mathrm{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\big(r_t(\theta)\hat{A}_t, \operatorname{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\big) \right]",
                    "minimind": "对应 trainer/train_ppo.py 第 203-216 行的 policy_loss 与 value_loss 双裁剪实现。"
                }
            ],
            "code_walkthrough": [
                ("trainer/train_ppo.py:L144-L151", "gen_len = old_resp_values.size(1); lastgaelam = torch.zeros(B, device=args.device); advs_rev = []\nfor t in reversed(range(gen_len)):\n    nv = old_resp_values[:, t + 1] if t < gen_len - 1 else 0.0\n    delta = token_rewards[:, t] + args.gamma * nv - old_resp_values[:, t]\n    lastgaelam = delta + args.gamma * args.lam * lastgaelam\n    advs_rev.append(lastgaelam)\nadvantages = torch.stack(advs_rev[::-1], dim=1)\nreturns = advantages + old_resp_values", "从最后一个生成位置倒序遍历到 0，计算单步 TD 误差 delta 与递推优势 lastgaelam，最后反转列表 stack 回正常时间顺序。")
            ],
            "hints": [
                "TODO(ch17-1) compute_gae：倒序 for t in reversed(range(T)): next_value = values[:, t + 1] if t < T - 1 else 0.0; delta = token_rewards[:, t] + gamma * next_value - values[:, t]; lastgaelam = delta + gamma * lam * lastgaelam; advs_rev.append(lastgaelam)",
                "TODO(ch17-2) ppo_policy_loss：ratio = torch.exp(logp - old_logp); loss = masked_mean(torch.max(-advantages * ratio, -advantages * torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)), mask); clipfrac = masked_mean(((ratio - 1.0).abs() > clip_eps).float(), mask)",
                "TODO(ch17-3) ppo_value_loss：v_clipped = torch.clamp(values, old_values - clip_value, old_values + clip_value); return 0.5 * masked_mean(torch.max((values - returns) ** 2, (v_clipped - returns) ** 2), mask)"
            ]
        },

        # ===================================================================== 16. 第 18 章
        {
            "id": "ch18",
            "stage": "阶段 5 · 强化学习对齐",
            "badge": "前沿重点",
            "difficulty": "Hard",
            "title": "第 18 章：DeepSeek GRPO 组内相对优势与 MiniMax CISPO 裁剪权重损失",
            "subtitle": "看懂为什么同一问题采样 G 条回答做组内标准化就能彻底省掉 Critic 模型，以及 CISPO 如何拯救反思转折词的梯度",
            "file": "grpo.py",
            "mm_path": "minimind/trainer/train_grpo.py#L121-L145",
            "visualizer": "rl_po",
            "todos": ["TODO(ch18-1)", "TODO(ch18-2)"],
            "theory_html": T(r"""
<h4>1. GRPO（Group Relative Policy Optimization）：用组内统计量彻底取代 Critic 模型</h4>
<p>在 PPO 中，训练一个与 Actor 同尺寸的 Critic 模型 $V_\phi$ 仅仅是为了给优势函数 $A_t = Q_t - V_\phi(s_t)$ 提供一个减方差的<strong>基线（Baseline）</strong>。DeepSeekMath (Shao et al., 2024) 提出：对于每一个输入问题 $q$，直接从当前旧策略采样一组 $G$ 个回答 $\{o_1, o_2, \dots, o_G\} \sim \pi_{\text{old}}(\cdot \mid q)$，获得标量奖励 $\{R_1, R_2, \dots, R_G\}$，直接用这 $G$ 个回答的<strong>组内均值与标准差</strong>构造基线：</p>
$$\hat{A}_{i, t} = \hat{A}_i = \frac{R_i - \operatorname{mean}(R_1, \dots, R_G)}{\operatorname{std}(R_1, \dots, R_G) + 10^{-4}}, \qquad \forall t \in \{1, \dots, |o_i|\}$$
<ul>
  <li><strong>显存直降近 50%</strong>：完全砍掉了 Critic 模型及其优化器状态，也不再需要计算 GAE 价值回归。</li>
  <li><strong>天然适配对比性质的奖励</strong>：无论题目本身简单（全组得分都在 2.5~3.0）还是困难（全组得分都在 -2.0~-1.5），组内标准化后总有约一半回答获得正优势、一半获得负优势，训练信号始终清晰稳定！</li>
</ul>

<h4>2. GRPO vs CISPO（Clipped Importance Sampling Policy Optimization，MiniMax-M1 2025）</h4>
<p>对比两者的 token 级目标函数（设概率比 $r_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\text{old}}(o_{i,t} \mid q, o_{i,<t})}$）：</p>
$$\mathcal{L}_{i,t}^{\mathrm{GRPO}}(\theta) = -\min\!\Big( r_{i,t}(\theta) \hat{A}_i,\; \operatorname{clip}(r_{i,t}(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_i \Big) + \beta \, \widehat{\mathrm{KL}}_{k_3}$$
$$\mathcal{L}_{i,t}^{\mathrm{CISPO}}(\theta) = -\operatorname{sg}\!\Big( \min\big(r_{i,t}(\theta),\, \epsilon_{\text{high}}\big) \Big) \cdot \hat{A}_i \cdot \log \pi_\theta(o_{i,t} \mid q, o_{i,<t}) + \beta \, \widehat{\mathrm{KL}}_{k_3}$$
<p><strong>为什么 CISPO 在长链推理上优于传统 PPO/GRPO Clip？</strong> 对上述两个损失关于 $\theta$ 求梯度（注意 $\nabla_\theta r_t = r_t \nabla_\theta \log \pi_\theta$）：</p>
$$\nabla_\theta \mathcal{L}_{i,t}^{\mathrm{GRPO}} = \begin{cases} -r_{i,t} \hat{A}_i \nabla_\theta \log \pi_\theta & \text{未触发裁剪} \\ \mathbf{0} \text{（梯度完全丢失！）} & (\hat{A}_i > 0 \wedge r_{i,t} > 1+\epsilon) \vee (\hat{A}_i < 0 \wedge r_{i,t} < 1-\epsilon) \end{cases}$$
$$\nabla_\theta \mathcal{L}_{i,t}^{\mathrm{CISPO}} = -\min\big(r_{i,t},\, \epsilon_{\text{high}}\big) \cdot \hat{A}_i \cdot \nabla_\theta \log \pi_\theta \qquad (\text{所有 token 始终保留非零梯度！})$$
<p>在推理过程中，决定思维链走向的<strong>分叉转折词</strong>（如 <code>"Wait"</code>、<code>"不过我们再检查一下"</code>）初始概率极低，一旦采样成功，其概率比 $r_{i,t}$ 极易在第一步更新后就突破 $1+\epsilon = 1.2$。GRPO 会直接在后续迭代中把这些最关键 token 的梯度砍成 0；而 CISPO 只把重要性权重封顶在 $\epsilon_{\text{high}} = 5.0$，让所有 token 都能持续参与策略更新！</p>
"""),
            "papers": [
                {
                    "title": "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO)",
                    "authors": "Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y.K. Li, Y. Wu, Daya Guo",
                    "venue": "arXiv:2402.03300 (2024)",
                    "pain": "PPO 必须训练一个与策略模型同等规模的 Value Critic，显存与算力开销翻倍，且在仅末尾有结果奖励的推理任务上 Value 难以准确拟合。",
                    "formula": r"\mathcal{J}_{\mathrm{GRPO}}(\theta) = \mathbb{E}_{q, \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}} \left[ \frac{1}{G}\sum_{i=1}^G \frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \left(\min(r_{i,t}\hat{A}_{i,t}, \operatorname{clip}(r_{i,t}, 1-\epsilon, 1+\epsilon)\hat{A}_{i,t}) - \beta \mathbb{D}_{\mathrm{KL}}\right) \right]",
                    "minimind": "对应 trainer/train_grpo.py 全部核心训练流程。"
                },
                {
                    "title": "MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention (CISPO)",
                    "authors": "MiniMax Team",
                    "venue": "arXiv:2506.13585 (2025)",
                    "pain": "PPO/GRPO 的 token-level ratio clipping 会把低概率高价值的反思分叉 token（rare fork tokens）在首次更新后直接剪掉梯度，阻碍长思维链涌现。",
                    "formula": r"\mathcal{J}_{\mathrm{CISPO}}(\theta) = \mathbb{E}\left[ \operatorname{sg}\!\big(\operatorname{clip}(r_{i,t}(\theta), 1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}})\big) \hat{A}_{i,t} \log \pi_\theta(o_{i,t} \mid q, o_{i,<t}) \right]",
                    "minimind": "对应 trainer/train_grpo.py 第 135-137 行（MiniMind 默认 loss_type='cispo'）。"
                }
            ],
            "code_walkthrough": [
                ("trainer/train_grpo.py:L121-L124", "grouped_rewards = rewards.view(-1, args.num_generations)\nmean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)\nstd_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)\nadvantages = (rewards - mean_r) / (std_r + 1e-4)", "4 行实现 GRPO 组内相对优势：view 成 [B, G] 求组内均值和总体标准差 (unbiased=False)，再用 repeat_interleave 展开回 [B*G]。"),
                ("trainer/train_grpo.py:L135-L143", "if args.loss_type == 'cispo':\n    clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()\n    per_token_loss = -(clamped_ratio * advantages.unsqueeze(1) * per_token_logps - args.beta * per_token_kl)\nelse:\n    clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)\n    per_token_loss1 = ratio * advantages.unsqueeze(1)\n    per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)\n    per_token_loss = -(torch.min(per_token_loss1, per_token_loss2) - args.beta * per_token_kl)\npolicy_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1)).mean()", "同时支持 cispo 与 grpo 两种损失，最后按序列长度求平均后再对 batch 求平均。")
            ],
            "hints": [
                "TODO(ch18-1) group_advantages：grouped = rewards.view(-1, num_generations); mean_r = grouped.mean(dim=1).repeat_interleave(num_generations); std_r = grouped.std(dim=1, unbiased=False).repeat_interleave(num_generations); return (rewards - mean_r) / (std_r + eps)",
                "TODO(ch18-2) grpo_loss：完全按照上方源码解析区分 loss_type == 'cispo' 与 'grpo'，最后返回 ((per_token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)).mean()。"
            ]
        },

        # ===================================================================== 17. 第 19-20 章
        {
            "id": "ch19",
            "stage": "阶段 6 · Agent 与前沿拓展",
            "badge": "前沿重点",
            "difficulty": "Medium",
            "title": "第 19–20 章：多轮 Tool-Call Agentic RL、可验证奖励 (RLVR) 与新架构拓展",
            "subtitle": "掌握多轮环境交互中的动作/环境轨迹掩码分离、AST 安全沙箱求值、离散扩散语言模型 (dLM) 与线性注意力",
            "file": "agent.py",
            "mm_path": "minimind/trainer/train_agent.py#L76-L244",
            "visualizer": "mask_vis",
            "todos": ["TODO(ch19-1)", "TODO(ch19-2)"],
            "theory_html": T(r"""
<h4>1. 多轮 Agentic RL 的轨迹概率分解与环境状态掩码（Trajectory Masking）</h4>
<p>在多轮工具调用智能体（ReAct / Agentic RL）中，一条包含 $K$ 轮交互的完整轨迹 $\tau$ 由<strong>模型生成的动作段 $a^{(k)}$</strong>（含 <code>&lt;think&gt;</code> 与 <code>&lt;tool_call&gt;</code> 或最终回答）和<strong>外部环境追加的观察段 $e^{(k)}$</strong>（即工具执行结果 <code>&lt;tool_response&gt;...&lt;/tool_response&gt;</code>）交替组成：</p>
$$\tau = \big( q,\; \underbrace{a^{(1)}}_{\text{mask}=1},\; \underbrace{e^{(1)}}_{\text{mask}=0},\; \underbrace{a^{(2)}}_{\text{mask}=1},\; \underbrace{e^{(2)}}_{\text{mask}=0},\; \dots,\; \underbrace{a^{(K)}}_{\text{mask}=1} \big)$$
<p>根据概率链式法则，整条轨迹生成概率为策略概率与环境转移概率的乘积：</p>
$$P_\theta(\tau \mid q) = \prod_{k=1}^{K} \pi_\theta\!\left(a^{(k)} \mid q, a^{(<k)}, e^{(<k)}\right) \cdot \prod_{k=1}^{K-1} P_{\text{env}}\!\left(e^{(k)} \mid a^{(k)}\right)$$
<p>由于环境转移概率 $P_{\text{env}}$ 与模型参数 $\theta$ 无关（$\nabla_\theta \log P_{\text{env}} \equiv \mathbf{0}$），对 $\log P_\theta(\tau \mid q)$ 求策略梯度时，<strong>只有模型自己生成的 token（$t \in \bigcup_k a^{(k)} \implies m_t = 1$）才参与计算策略损失和 KL 惩罚，而环境注入的所有 token 必须严格设 $m_t = 0$</strong>！</p>

<h4>2. 可验证奖励强化学习（RLVR）的多维奖励塑形</h4>
<p>MiniMind 在 <code>train_agent.py</code> 中采用完全无需人工标注的多维自动化可验证奖励 $R(\tau) \in [-3, 3]$：</p>
$$R(\tau) = \operatorname{clip}\!\left( R_{\text{format}} + R_{\text{tool\_align}} + R_{\text{GT}} - R_{\text{unfinished}} - R_{\text{rep}},\; -3.0,\; 3.0 \right)$$
<ul>
  <li><strong>标签闭合惩罚 $R_{\text{format}}$</strong>：每出现一次 <code>&lt;tool_call&gt;</code> 与 <code>&lt;/tool_call&gt;</code> 数量不相等，扣 $0.5$ 分。</li>
  <li><strong>调用次数与参数合法性对齐分 $R_{\text{tool\_align}}$</strong>：记合法工具调用数为 $C_{\text{valid}}$，总调用数为 $C_{\text{total}}$，标准答案数为 $|\text{GT}|$，差值 $\Delta_{\text{tool}} = |C_{\text{valid}} - |\text{GT}|| + (C_{\text{total}} - C_{\text{valid}})$；当 $\Delta_{\text{tool}} = 0$ 时奖 $+0.5$，否则罚 $-0.5 \Delta_{\text{tool}}$。</li>
  <li><strong>最终答案 Ground Truth 校验分 $R_{\text{GT}}$</strong>：按最终回复命中标准答案的比例奖 $+2.5 \times \frac{|\text{verified}|}{|\text{GT}|}$。</li>
</ul>

<h4>3. 前沿架构拓展：离散扩散语言模型（dLM）与 $O(Ld^2)$ 线性注意力</h4>
<ul>
  <li><strong>离散扩散语言模型（dLM，Discussion #618 / LLaDA 2025）</strong>：去掉单向因果掩码（改为双向注意力），前向加噪过程按比例 $t \sim \mathcal{U}(0, 1)$ 随机将回复 token 替换为 <code>[MASK]</code>，训练目标为仅在被掩码位置预测原词：
    $$\mathcal{L}_{\mathrm{dLM}}(\theta) = -\mathbb{E}_{t \sim \mathcal{U}(0,1),\, \tilde{\mathbf{y}} \sim q_{t|0}(\cdot|\mathbf{y})} \left[ \frac{1}{t} \sum_{i=1}^{L} \mathbb{I}(\tilde{y}_i = \text{[MASK]}) \log p_\theta(y_i \mid \mathbf{x}, \tilde{\mathbf{y}}) \right]$$
    推理时从全 <code>[MASK]</code> 出发，每步并行预测全部位置并锁定高置信度 token，实现<strong>数十步并行生成整段文本</strong>！</li>
  <li><strong>线性注意力（Linear Attention，Discussion #704）</strong>：用核特征映射 $\phi(\cdot)$ 改写注意力并<strong>利用矩阵乘法结合律先算 $\phi(K)^T V$</strong>：
    $$\operatorname{Attn}(Q, K, V) = \underbrace{\big(\phi(Q) \phi(K)^T\big)}_{L \times L \text{ 矩阵，} O(L^2 d)} V = \phi(Q) \underbrace{\big(\phi(K)^T V\big)}_{d \times d \text{ 状态矩阵，} O(L d^2)}$$
    复杂度对序列长度 $L$ 从二次方降为严格线性 $O(L d^2)$，且推理时仅需维护恒定大小的前缀状态矩阵 $\mathbf{S}_t = \mathbf{S}_{t-1} + \phi(\mathbf{k}_t)\mathbf{v}_t^T \in \mathbb{R}^{d \times d}$！</li>
</ul>
"""),
            "papers": [
                {
                    "title": "ReAct: Synergizing Reasoning and Acting in Language Models & Toolformer",
                    "authors": "Shunyu Yao et al. (ICLR 2023 · arXiv:2210.03629) / Timo Schick et al. (NeurIPS 2023 · arXiv:2302.04761)",
                    "pain": "纯语言模型的内部参数知识无法实时更新，且在多位数算术运算和精确事实检索上容易产生幻觉。",
                    "formula": r"\text{Thought } \tau_t \to \text{Action (Tool Call) } a_t \to \text{Observation (Tool Response) } o_t \to \text{Thought } \tau_{t+1}",
                    "minimind": "对应 trainer/train_agent.py 与 scripts/eval_toolcall.py 的多轮工具交互闭环。"
                },
                {
                    "title": "LLaDA: Large Language Diffusion Models & Transformers are RNNs (Linear Attention)",
                    "authors": "Shen Nie et al. (2025 · arXiv:2502.09992) / Angelos Katharopoulos et al. (ICML 2020 · arXiv:2006.16236)",
                    "venue": "arXiv 2025 / ICML 2020",
                    "pain": "自回归模型受限于逐 token 串行解码与 O(L²) 的 KV Cache 增长。",
                    "formula": r"\mathbf{S}_t = \mathbf{S}_{t-1} + \phi(\mathbf{k}_t)\mathbf{v}_t^T, \quad \mathbf{y}_t = \frac{\phi(\mathbf{q}_t)^T \mathbf{S}_t}{\phi(\mathbf{q}_t)^T \mathbf{z}_t}",
                    "minimind": "对应 MiniMind Discussions #618 (dLM) 与 #704 (Linear Attention) 实验性分支。"
                }
            ],
            "code_walkthrough": [
                ("trainer/train_agent.py:L121-L123", "response_ids.extend(new_ids)\nresponse_mask.extend([int(t != tokenizer.eos_token_id) for t in new_ids])\nresponse_old_logps.extend(new_logps)", "和上方的工具观察对比，这里缓存模型生成的 token 及其旧策略对数概率。非 EOS 的生成位置参与策略更新。"),
                ("trainer/trainer_utils.py:L184-L186", "def pow_guard(base, exp):\n    if abs(exp) > 1e4 or (abs(base) > 1 and abs(exp) * math.log10(abs(base)) > 100): raise ValueError('幂运算结果过大')\n    return base ** exp", "AST 安全求值沙箱：拦截 9**9**9 等超大指数运算，防止模型在 RL 探索中生成算力炸弹把训练进程卡死。")
            ],
            "hints": [
                "TODO(ch19-1) parse_tool_calls：遍历 for m in TOOL_CALL_RE.findall(text): 在 try 块中 calls.append(json.loads(m.strip()))，异常则 pass。",
                "TODO(ch19-2) tool_reward：按注释依次扣减标签未闭合分、加减合法调用对齐分、加上 2.5 * len(validate_gt_in_text(final_answer, gt)) / len(gt)，处理 unfinished，最后 clip 到 [-3.0, 3.0]。"
            ]
        },
    ]

    for ch in chapters:
        lab_code, ref_code = read_code_pair(ch["file"])
        ch["lab_code"] = lab_code
        ch["ref_code"] = ref_code

    load_guides(chapters)
    return chapters


def build_html():
    chapters = build_chapters_data()
    katex_css = (ASSETS_DIR / "katex.min.css").read_text(encoding="utf-8")
    def _inline_font(m):
        fname = m.group(1)
        fp = ASSETS_DIR / "fonts" / fname
        if fp.exists():
            b64 = base64.b64encode(fp.read_bytes()).decode("ascii")
            return f"url(data:font/woff2;base64,{b64}) format(\"woff2\")"
        return m.group(0)
    katex_css = re.sub(
        r"url\(fonts/([A-Za-z0-9_\-]+\.woff2)\) format\(\"woff2\"\),url\(fonts/[^)]+\.woff\) format\(\"woff\"\),url\(fonts/[^)]+\.ttf\) format\(\"truetype\"\)",
        _inline_font,
        katex_css,
    )
    katex_js = (ASSETS_DIR / "katex.min.js").read_text(encoding="utf-8")

    chapters_json = json.dumps(chapters, ensure_ascii=False).replace("</script>", "<\\/script>")

    html_doc = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=5.0" />
<meta name="apple-mobile-web-app-capable" content="yes" />
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent" />
<meta name="theme-color" content="#3b82f6" />
<link rel="manifest" href="manifest.json" />
<title>Learn-MiniMind：交互式 LLM 底层原理、论文精读与 LeetCode 在线编程主页</title>
<style>__KATEX_CSS__</style>
<style>
:root {
  --background: #0b0d12;
  --foreground: #e8ecf4;
  --muted: #161b26;
  --muted-foreground: #94a3b8;
  --card: #121620;
  --card-foreground: #e8ecf4;
  --popover: #161b26;
  --secondary: #1e2636;
  --secondary-foreground: #e2e8f0;
  --border: #263044;
  --ring: #3b82f6;
  --primary: #3b82f6;
  --primary-foreground: #ffffff;
  --accent: #6366f1;
  --accent-surface: rgba(99, 102, 241, 0.14);
  --destructive: #ef4444;
  --destructive-surface: rgba(239, 68, 68, 0.14);
  --warning: #f59e0b;
  --warning-surface: rgba(245, 158, 11, 0.14);
  --success: #10b981;
  --info: #06b6d4;
  --code-background: #0d1017;
  --chart-1: #3b82f6;
  --chart-2: #10b981;
  --chart-3: #f59e0b;
  --chart-4: #a855f7;
  --chart-5: #ec4899;
  --chart-6: #06b6d4;
  --radius: 10px;
  --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
  --font-mono: "JetBrains Mono", "Fira Code", "SF Mono", Menlo, Monaco, Consolas, monospace;
}
[data-theme="light"] {
  --background: #f8fafc;
  --foreground: #0f172a;
  --muted: #f1f5f9;
  --muted-foreground: #475569;
  --card: #ffffff;
  --card-foreground: #0f172a;
  --popover: #ffffff;
  --secondary: #e2e8f0;
  --secondary-foreground: #1e293b;
  --border: #cbd5e1;
  --ring: #2563eb;
  --primary: #2563eb;
  --primary-foreground: #ffffff;
  --accent: #4f46e5;
  --accent-surface: rgba(79, 70, 229, 0.10);
  --code-background: #f1f5f9;
}
.sync-drawer {
  margin-top: 12px;
  padding: 14px;
  background: var(--muted);
  border: 1px solid var(--primary);
  border-radius: calc(var(--radius) - 2px);
  font-size: 12.5px;
}
.sync-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 10px;
  margin-top: 10px;
}
.sync-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 10px 12px;
}
.sync-input {
  width: 100%;
  font-family: var(--font-mono);
  font-size: 11.5px;
  padding: 6px 8px;
  margin: 4px 0;
  border-radius: 5px;
  border: 1px solid var(--border);
  background: var(--code-background);
  color: var(--foreground);
}
* { box-sizing: border-box; }
html, body {
  margin: 0;
  padding: 0;
  color: var(--foreground);
  font-family: var(--font-sans);
  line-height: 1.6;
}
.app-root {
  width: 100%;
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.box {
  background: var(--card);
  color: var(--card-foreground);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px;
}
.topbar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}
.brand-title {
  font-size: 18px;
  font-weight: 700;
  letter-spacing: -0.01em;
  display: flex;
  align-items: center;
  gap: 8px;
}
.brand-sub {
  font-size: 12px;
  color: var(--muted-foreground);
  margin-top: 2px;
}
.status-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
}
.pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  font-weight: 600;
  padding: 4px 10px;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: var(--secondary);
  color: var(--secondary-foreground);
}
.pill-success {
  border-color: var(--success);
  color: var(--success);
}
.pill-accent {
  background: var(--accent-surface);
  border-color: var(--accent);
  color: var(--primary);
}
.chapter-Strip {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.ch-btn {
  cursor: pointer;
  font-family: var(--font-sans);
  font-size: 12px;
  font-weight: 600;
  padding: 6px 10px;
  border-radius: calc(var(--radius) - 3px);
  border: 1px solid var(--border);
  background: var(--muted);
  color: var(--foreground);
  transition: all 0.15s ease;
  display: inline-flex;
  align-items: center;
  gap: 5px;
}
.ch-btn:hover {
  border-color: var(--primary);
}
.ch-btn.active {
  background: var(--primary);
  color: var(--primary-foreground);
  border-color: var(--primary);
}
.ch-btn.solved {
  border-color: var(--success);
}
.ch-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--muted-foreground);
}
.ch-btn.solved .ch-dot {
  background: var(--success);
}
.ch-btn.active .ch-dot {
  background: var(--primary-foreground);
}
.tab-bar {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  border-bottom: 1px solid var(--border);
  padding-bottom: 10px;
  margin-bottom: 14px;
}
.tab-btn {
  cursor: pointer;
  font-family: var(--font-sans);
  font-size: 12.5px;
  font-weight: 600;
  padding: 6px 12px;
  border-radius: calc(var(--radius) - 3px);
  border: 1px solid var(--border);
  background: var(--muted);
  color: var(--muted-foreground);
}
.tab-btn.active {
  background: var(--accent-surface);
  color: var(--primary);
  border-color: var(--primary);
}
.ch-header-title {
  font-size: 17px;
  font-weight: 700;
  margin: 0 0 4px 0;
}
.ch-header-sub {
  font-size: 12.5px;
  color: var(--muted-foreground);
  margin: 0 0 12px 0;
}
.theory-body h4 {
  font-size: 14px;
  margin: 12px 0 6px 0;
  color: var(--primary);
  border-left: 3px solid var(--primary);
  padding-left: 8px;
}
.theory-body h4:first-child {
  margin-top: 2px;
}
.theory-body p, .theory-body li {
  font-size: 13px;
  line-height: 1.62;
}
.theory-body ul, .theory-body ol {
  padding-left: 20px;
  margin: 6px 0;
}
.theory-body code {
  font-family: var(--font-mono);
  font-size: 12px;
  background: var(--code-background);
  padding: 2px 5px;
  border-radius: 4px;
  border: 1px solid var(--border);
}
.theory-body pre {
  background: var(--code-background);
  padding: 10px 12px;
  border-radius: calc(var(--radius) - 2px);
  border: 1px solid var(--border);
  overflow-x: auto;
  font-size: 12px;
}
.math-block {
  margin: 8px 0;
  padding: 10px 12px;
  background: var(--code-background);
  border: 1px solid var(--border);
  border-radius: calc(var(--radius) - 2px);
  overflow-x: auto;
  text-align: center;
}
.math-block .katex-display {
  margin: 0;
  padding: 4px 0;
  overflow-x: auto;
  overflow-y: hidden;
}
.math-inline {
  padding: 0 2px;
}
.paper-card {
  background: var(--muted);
  border: 1px solid var(--border);
  border-radius: calc(var(--radius) - 2px);
  padding: 14px;
  margin-bottom: 12px;
}
.paper-title {
  font-size: 14px;
  font-weight: 700;
  color: var(--primary);
  margin-bottom: 4px;
}
.paper-meta {
  font-size: 12px;
  color: var(--muted-foreground);
  margin-bottom: 8px;
}
.paper-row {
  font-size: 13px;
  margin: 6px 0;
}
.code-walk-card {
  background: var(--code-background);
  border: 1px solid var(--border);
  border-radius: calc(var(--radius) - 2px);
  padding: 12px;
  margin-bottom: 10px;
}
.code-walk-loc {
  font-family: var(--font-mono);
  font-size: 12px;
  font-weight: 700;
  color: var(--warning);
  margin-bottom: 6px;
}
.code-walk-pre {
  font-family: var(--font-mono);
  font-size: 12px;
  white-space: pre-wrap;
  word-break: break-word;
  background: var(--card);
  padding: 8px 10px;
  border-radius: 6px;
  border: 1px solid var(--border);
  margin: 6px 0;
}
.code-walk-desc {
  font-size: 12.5px;
  color: var(--muted-foreground);
}
.ide-toolbar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  margin-bottom: 10px;
}
.btn {
  cursor: pointer;
  font-family: var(--font-sans);
  font-size: 12.5px;
  font-weight: 600;
  padding: 7px 12px;
  border-radius: calc(var(--radius) - 3px);
  border: 1px solid var(--border);
  background: var(--secondary);
  color: var(--secondary-foreground);
  display: inline-flex;
  align-items: center;
  gap: 5px;
}
.btn:hover {
  border-color: var(--primary);
}
.btn-primary {
  background: var(--primary);
  color: var(--primary-foreground);
  border-color: var(--primary);
}
.btn-success {
  background: var(--success);
  color: #ffffff;
  border-color: var(--success);
}
.code-editor {
  width: 100%;
  font-family: var(--font-mono);
  font-size: 12.5px;
  line-height: 1.55;
  background: var(--code-background);
  color: var(--foreground);
  border: 1px solid var(--border);
  border-radius: calc(var(--radius) - 2px);
  padding: 12px;
  resize: none;
  tab-size: 4;
}
.code-editor:focus {
  outline: 2px solid var(--ring);
}
.todo-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-bottom: 10px;
}
.todo-chip {
  cursor: pointer;
  font-family: var(--font-mono);
  font-size: 11.5px;
  padding: 3px 8px;
  border-radius: 6px;
  background: var(--warning-surface);
  color: var(--warning);
  border: 1px solid var(--warning);
}
.todo-chip.done {
  background: rgba(16, 185, 129, 0.14);
  color: var(--success);
  border-color: var(--success);
}
.console-box {
  margin-top: 12px;
  background: var(--code-background);
  border: 1px solid var(--border);
  border-radius: calc(var(--radius) - 2px);
  padding: 12px;
}
.testcase-card {
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 8px 10px;
  margin-top: 8px;
  background: var(--card);
  font-size: 12.5px;
}
.testcase-pass {
  border-left: 4px solid var(--success);
}
.testcase-fail {
  border-left: 4px solid var(--destructive);
}
.vis-panel {
  background: var(--muted);
  border: 1px solid var(--border);
  border-radius: calc(var(--radius) - 2px);
  padding: 14px;
}
.vis-controls {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 10px;
  margin-bottom: 12px;
}
.vis-ctrl {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 8px 10px;
  font-size: 12px;
}
.vis-ctrl input[type="range"] {
  width: 100%;
  margin-top: 4px;
}
.metric-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(135px, 1fr));
  gap: 8px;
  margin: 10px 0;
}
.metric-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 8px 10px;
}
.metric-label {
  font-size: 11px;
  color: var(--muted-foreground);
}
.metric-val {
  font-family: var(--font-mono);
  font-size: 15px;
  font-weight: 700;
  color: var(--primary);
  margin-top: 2px;
}
.bar-row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 5px 0;
  font-size: 12px;
  font-family: var(--font-mono);
}
.bar-track {
  flex: 1;
  height: 16px;
  background: var(--code-background);
  border-radius: 4px;
  overflow: hidden;
  border: 1px solid var(--border);
}
.bar-fill {
  height: 16px;
  background: var(--chart-1);
  transition: width 0.2s ease;
}
.token-badge {
  display: inline-block;
  font-family: var(--font-mono);
  font-size: 12px;
  padding: 3px 7px;
  margin: 3px;
  border-radius: 5px;
  border: 1px solid var(--border);
}
.token-active {
  background: rgba(16, 185, 129, 0.18);
  border-color: var(--success);
  color: var(--success);
}
.token-masked {
  background: var(--code-background);
  color: var(--muted-foreground);
  opacity: 0.7;
}
</style>
<style>__WORKSPACE_CSS__</style>
</head>
<body>
<div class="app-root">
  <!-- 顶部导航与状态栏 -->
  <header class="box course-header">
    <div class="topbar">
      <h1 class="brand-title">Learn-MiniMind <small>从零手写语言模型</small></h1>
      <div class="status-row">
        <span id="progress-pill" class="pill pill-accent"></span>
        <button class="btn" onclick="toggleSyncDrawer()" aria-controls="sync-drawer">多端同步</button>
        <button class="btn" onclick="toggleTheme()" id="theme-toggle-btn">切换主题</button>
      </div>
    </div>
    <div class="course-controls">
      <nav aria-label="课程章节" class="chapter-Strip" id="chapter-nav"></nav>
      <label class="chapter-picker">章节 <select id="chapter-select" onchange="openLesson(this.value)"></select></label>
      <div class="view-switch" aria-label="工作区布局" role="group">
        <button data-view="learn" onclick="setView('learn')">教材</button>
        <button data-view="split" onclick="setView('split')">分栏</button>
        <button data-view="code" onclick="setView('code')">代码</button>
      </div>
    </div>

    <!-- 多设备端同步中心抽屉 -->
    <div id="sync-drawer" class="sync-drawer" style="display:none;">
      <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
        <div>
          <strong style="font-size:13.5px; color:var(--primary);">☁️ 多设备学习进度与代码同步中心（电脑 · iPad · 手机无缝衔接）</strong>
          <span style="font-size:11.5px; color:var(--muted-foreground); margin-left:8px;" id="sync-status-msg">当前设备所有代码修改与闯关进度均已自动保存在浏览器本地 (localStorage)</span>
        </div>
        <button class="btn" onclick="toggleSyncDrawer()">✕ 收起</button>
      </div>
      <div class="sync-grid">
        <div class="sync-card">
          <div style="font-weight:700; margin-bottom:4px;">① 免登录跨设备同步链接 / 同步码</div>
          <div style="font-size:11.5px; color:var(--muted-foreground); margin-bottom:6px;">将你的通关进度与当前代码打包为带 <code>#sync=...</code> 的网址或同步码，发到手机/iPad 打开即刻恢复：</div>
          <div style="display:flex; gap:6px; flex-wrap:wrap;">
            <button class="btn btn-primary" onclick="copySyncUrl()">🔗 复制跨设备同步链接</button>
            <button class="btn" onclick="importFromSyncCodePrompt()">📋 粘贴同步码恢复</button>
          </div>
          <input type="text" id="sync-url-preview" class="sync-input" readonly placeholder="点击上方按钮生成跨设备同步链接..." style="margin-top:6px;" />
        </div>

        <div class="sync-card">
          <div style="font-weight:700; margin-bottom:4px;">② 完整代码存档 (.json) 导出 / 导入</div>
          <div style="font-size:11.5px; color:var(--muted-foreground); margin-bottom:6px;">将全部 18 个章节的自定义代码与通关记录备份为本地 JSON 文件，随时在任意设备上还原：</div>
          <div style="display:flex; gap:6px; flex-wrap:wrap;">
            <button class="btn" onclick="exportProgressJson()">📥 导出学习存档 (.json)</button>
            <button class="btn" onclick="document.getElementById('sync-file-input').click()">📤 导入存档文件</button>
            <input type="file" id="sync-file-input" accept=".json" style="display:none;" onchange="importProgressJson(event)" />
          </div>
        </div>

        <div class="sync-card">
          <div style="font-weight:700; margin-bottom:4px;">③ GitHub Gist 云端实时同步</div>
          <div style="font-size:11.5px; color:var(--muted-foreground); margin-bottom:4px;">填入具有 <code>gist</code> 权限的 GitHub Token（仅保存在你当前浏览器本地），一键跨端云同步：</div>
          <input type="password" id="gist-token-input" class="sync-input" placeholder="ghp_xxx (GitHub Personal Access Token)" onchange="saveGistConfig()" />
          <input type="text" id="gist-id-input" class="sync-input" placeholder="Gist ID (首次上传留空将自动创建)" onchange="saveGistConfig()" />
          <div style="display:flex; gap:6px; margin-top:4px;">
            <button class="btn btn-primary" onclick="pushToGist()">☁️ 上传到云端</button>
            <button class="btn" onclick="pullFromGist()">🔄 从云端拉取</button>
          </div>
        </div>
      </div>
    </div>
  </header>

  <main class="workspace-grid" id="workspace" data-desktop-view="split" data-mobile-view="learn">
    <section class="box study-pane" id="pane-learn" aria-labelledby="ch-title">
      <div class="lesson-heading">
        <div id="ch-meta-bar"></div>
        <h2 class="ch-header-title" id="ch-title"></h2>
      </div>
      <nav class="tab-bar" aria-label="本章学习内容">
        <button class="tab-btn active" onclick="switchTab('theory')" id="tab-btn-theory">导读与原理</button>
        <button class="tab-btn" onclick="switchTab('vis')" id="tab-btn-vis">可视化实验</button>
        <button class="tab-btn" onclick="switchTab('walk')" id="tab-btn-walk">源码对照</button>
        <button class="tab-btn" onclick="switchTab('papers')" id="tab-btn-papers">论文精读</button>
      </nav>

      <div class="lesson-scroll" id="lesson-scroll" tabindex="0" aria-label="本章教材">
        <div id="tab-content-theory">
          <div id="lesson-guide"></div>
          <h3 class="derivation-title">原理与推导 <span id="latex-status-pill" class="pill pill-success">公式自检中</span></h3>
          <div class="theory-body" id="theory-text"></div>
        </div>
        <div id="tab-content-papers" hidden></div>
        <div id="tab-content-vis" hidden></div>
        <div id="tab-content-walk" hidden></div>
      </div>
    </section>

    <div class="split-handle width-handle" id="width-handle" role="separator" tabindex="0"
         aria-orientation="vertical" aria-controls="pane-learn" aria-label="教材宽度"
         aria-valuemin="30" aria-valuemax="70" aria-valuenow="54"
         title="左右拖动调整分栏；方向键微调，双击恢复默认"></div>

    <section class="box study-pane" id="pane-code" aria-labelledby="ide-file-title">
      <div class="ide-toolbar">
        <h2 id="ide-file-title">代码练习</h2>
        <div class="grade-actions">
          <button id="backend-status-pill" class="pill" onclick="checkBackend(true)" title="连接本地 PyTorch 评测服务">浏览器规则检查</button>
          <button class="btn btn-primary" onclick="submitAndGrade()">运行评测</button>
        </div>
      </div>
      <div class="exercise-tools">
        <div class="todo-chips" id="todo-chips-bar" aria-label="定位练习"></div>
        <div class="editor-actions">
          <button class="btn" onclick="toggleHint()" aria-controls="hint-drawer">提示</button>
          <button class="btn" onclick="fillSolution()">填入参考答案</button>
          <button class="btn" onclick="resetCurrentCode()">重置代码</button>
        </div>
      </div>
      <div id="hint-drawer" hidden></div>
      <div class="editor-stack" id="editor-stack">
        <textarea id="code-editor" class="code-editor" wrap="off" aria-label="Python 练习代码" spellcheck="false" autocapitalize="off" autocomplete="off"></textarea>
        <div class="split-handle height-handle" id="height-handle" role="separator" tabindex="0"
             aria-orientation="horizontal" aria-controls="code-editor" aria-label="代码编辑区高度"
             aria-valuemin="40" aria-valuemax="100" aria-valuenow="100"
             title="上下拖动调整编辑器与评测区；方向键微调，双击恢复默认"></div>
        <section class="console-box" id="grade-console">
          <button class="console-toggle" id="console-toggle" aria-expanded="false" aria-controls="testcase-results" onclick="toggleConsole()">
            <strong>评测结果</strong><span id="console-status-text" aria-live="polite">待运行</span><span class="disclosure" aria-hidden="true">⌃</span>
          </button>
          <div id="testcase-results" tabindex="0" hidden></div>
        </section>
      </div>
    </section>
  </main>
</div>

<script>__KATEX_JS__</script>
<script>
const CHAPTERS = __CHAPTERS_JSON__;
let currentChapterIdx = 0;
let currentTab = 'theory';
let backendAvailable = false;
let backendUrl = 'http://127.0.0.1:8765';
const userCodeStore = {};
const solvedStore = {};
let displayedChapterId = null;
let gradeRequest = 0;

__WORKSPACE_JS__

// ==================== 1. LaTeX 零故障渲染与严格自检 ====================
function renderAllMath(container) {
  const root = container || document;
  const nodes = root.querySelectorAll('[data-tex]');
  let renderedCount = 0;
  let errorCount = 0;
  nodes.forEach(node => {
    const tex = node.getAttribute('data-tex');
    const isDisplay = node.classList.contains('math-block');
    try {
      katex.render(tex, node, {
        displayMode: isDisplay,
        throwOnError: true,
        strict: false
      });
      renderedCount++;
    } catch (err) {
      errorCount++;
      console.error('KaTeX Error on:', tex, err);
      node.textContent = '[LaTeX Error: ' + tex + ']';
    }
  });
  return { renderedCount, errorCount };
}

function verifyAllChaptersLatex() {
  let totalFormulas = 0;
  let totalErrors = 0;
  const scratch = document.createElement('div');
  CHAPTERS.forEach(ch => {
    scratch.innerHTML = ch.theory_html;
    ch.papers.forEach(p => {
      const d = document.createElement('div');
      d.className = 'math-block';
      d.setAttribute('data-tex', p.formula);
      scratch.appendChild(d);
    });
    const res = renderAllMath(scratch);
    totalFormulas += res.renderedCount;
    totalErrors += res.errorCount;
  });
  const pill = document.getElementById('latex-status-pill');
  if (totalErrors === 0) {
    pill.textContent = `公式 ✓ ${totalFormulas}`;
    pill.title = `${totalFormulas} 个公式渲染通过，0 个错误`;
    pill.className = 'pill pill-success';
  } else {
    pill.textContent = `⚠️ LaTeX 公式发现 ${totalErrors} 个错误`;
  }
}

// ==================== 2. 检测本地 PyTorch 评测后端 ====================
async function checkBackend(manual) {
  if (!manual && window.location.port !== '8765') return;
  try {
    const res = await fetch(backendUrl + '/api/health', { method: 'GET' });
    if (res.ok) {
      const data = await res.json();
      backendAvailable = true;
      const bPill = document.getElementById('backend-status-pill');
      bPill.textContent = `🟢 本地 PyTorch ${data.torch} 内核已连接`;
      bPill.className = 'pill pill-success';
    }
  } catch (e) {
    backendAvailable = false;
  }
}

// ==================== 3. 章节导航与标签页切换 ====================
function renderChapterNav() {
  const nav = document.getElementById('chapter-nav');
  const picker = document.getElementById('chapter-select');
  nav.replaceChildren();
  picker.replaceChildren();
  CHAPTERS.forEach((ch, idx) => {
    const active = idx === currentChapterIdx;
    const solved = !!solvedStore[ch.id];
    const btn = document.createElement('button');
    btn.className = 'ch-btn' + (active ? ' active' : '') + (solved ? ' solved' : '');
    btn.innerHTML = `<span class="ch-key">${chapterNumber(ch)}</span> ${escapeHtml(ch.guide.nav)}`;
    btn.title = ch.title;
    if (active) btn.setAttribute('aria-current', 'step');
    btn.onclick = () => openLesson(ch.id);
    nav.appendChild(btn);
    picker.add(new Option(chapterNumber(ch) + ' · ' + ch.guide.nav + (solved ? ' ✓' : ''), ch.id));
  });
  picker.value = CHAPTERS[currentChapterIdx].id;
  const solvedCount = CHAPTERS.filter(ch => solvedStore[ch.id]).length;
  document.getElementById('progress-pill').textContent = `进度 ${solvedCount} / ${CHAPTERS.length}`;
}

function selectChapter(idx, captureDraft = true) {
  const editor = document.getElementById('code-editor');
  if (captureDraft && displayedChapterId) userCodeStore[displayedChapterId] = editor.value;
  currentChapterIdx = idx;
  const ch = CHAPTERS[idx];
  displayedChapterId = ch.id;
  gradeRequest++;

  document.getElementById('ch-meta-bar').textContent = ch.stage + '　/　第 ' + chapterNumber(ch) + ' 章';
  const heading = document.getElementById('ch-title');
  heading.textContent = ch.guide.topic;
  heading.title = ch.title;
  renderGuide(ch);

  // 1) 渲染理论与公式
  const theoryEl = document.getElementById('theory-text');
  theoryEl.innerHTML = ch.theory_html;
  renderAllMath(theoryEl);

  // 2) 渲染论文精读卡片
  const papersEl = document.getElementById('tab-content-papers');
  papersEl.innerHTML = ch.papers.map((p, i) => `
    <div class="paper-card">
      <div class="paper-title">📄 [论文 ${i + 1}] ${p.title}</div>
      <div class="paper-meta">作者: ${p.authors} ｜ 发表: ${p.venue}</div>
      <div class="paper-row"><strong>🎯 解决的核心底层痛点：</strong>${p.pain}</div>
      <div class="paper-row"><strong>📐 论文核心数学公式：</strong></div>
      <div class="math-block" data-tex="${p.formula.replace(/"/g, '&quot;')}"></div>
      <div class="paper-row"><strong>🔗 MiniMind 源码落地位置：</strong>${p.minimind}</div>
    </div>
  `).join('');
  renderAllMath(papersEl);

  // 3) 渲染交互式可视化实验室
  renderVisualizer(ch.visualizer);
  if (ch.guide.practice[0] === 'vis') {
    const instruction = document.createElement('p');
    instruction.className = 'guide-task';
    instruction.textContent = ch.guide.practice[1];
    document.getElementById('tab-content-vis').prepend(instruction);
  }

  // 4) 渲染源码逐行对照
  const walkEl = document.getElementById('tab-content-walk');
  walkEl.innerHTML = renderSourceExcerpt(ch.guide.source) + ch.code_walkthrough.map(([loc, snippet, desc]) => `
    <div class="code-walk-card">
      <div class="code-walk-loc">📍 ${loc}</div>
      <div class="code-walk-pre">${escapeHtml(snippet)}</div>
      <div class="code-walk-desc">💡 <strong>底层解析：</strong>${escapeHtml(desc)}</div>
    </div>
  `).join('');

  // 5) 渲染右侧 IDE
  document.getElementById('ide-file-title').textContent = 'labs/' + ch.file;
  editor.value = userCodeStore[ch.id] ?? ch.lab_code;
  editor.setSelectionRange(0, 0);
  editor.scrollTop = editor.scrollLeft = 0;
  document.getElementById('hint-drawer').hidden = true;
  document.querySelector('[aria-controls="hint-drawer"]').setAttribute('aria-expanded', false);
  document.getElementById('hint-drawer').innerHTML = ch.hints.map(h => `<div>• ${escapeHtml(h)}</div>`).join('');
  document.getElementById('console-status-text').textContent = '待运行';
  document.getElementById('testcase-results').textContent = '完成本章练习后点「运行评测」查看结果。';
  updateTodoChips();
  renderChapterNav();
  switchTab('theory');
  saveLocalState();
}

function switchTab(tabName) {
  currentTab = tabName;
  ['theory', 'papers', 'vis', 'walk'].forEach(t => {
    const active = t === tabName;
    document.getElementById('tab-content-' + t).hidden = !active;
    const button = document.getElementById('tab-btn-' + t);
    button.classList.toggle('active', active);
    button.setAttribute('aria-pressed', active);
  });
  document.getElementById('lesson-scroll').scrollTop = 0;
}

function escapeHtml(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// ==================== 4. 右侧 LeetCode 编辑器与双引擎评测 ====================
function updateTodoChips() {
  const ch = CHAPTERS[currentChapterIdx];
  const code = document.getElementById('code-editor').value;
  const bar = document.getElementById('todo-chips-bar');
  bar.innerHTML = ch.todos.map(todo => {
    const tag = todo.match(/ch\\d+-\\d+/)[0];
    const notImplRe = new RegExp(`raise\\\\s+NotImplementedError\\\\(["']${tag}`);
    const isDone = !notImplRe.test(code);
    return `<button class="todo-chip ${isDone ? 'done' : ''}" title="定位到 ${todo}" onclick="jumpToTodo('${todo}')">${isDone ? '✓ ' : ''}${tag}</button>`;
  }).join('');
}

function jumpToTodo(todo) {
  const editor = document.getElementById('code-editor');
  const ch = CHAPTERS[currentChapterIdx];
  let needle = todo;
  let idx = editor.value.indexOf(todo);
  // A learner may have deleted the TODO comment. Locate its original function instead.
  if (idx < 0) {
    const prefix = ch.lab_code.slice(0, ch.lab_code.indexOf(todo));
    const defs = [...prefix.matchAll(/^ *def (\\w+)\\(/gm)];
    if (defs.length) {
      needle = 'def ' + defs[defs.length - 1][1] + '(';
      idx = editor.value.indexOf(needle);
    }
  }
  editor.focus({ preventScroll: true });
  if (idx >= 0) {
    editor.setSelectionRange(idx, idx + needle.length);
    const line = editor.value.slice(0, idx).split('\\n').length - 1;
    editor.scrollTop = Math.max(0, line * parseFloat(getComputedStyle(editor).lineHeight) - editor.clientHeight / 3);
    editor.scrollLeft = 0;
  }
}

function toggleHint() {
  const drawer = document.getElementById('hint-drawer');
  drawer.hidden = !drawer.hidden;
  document.querySelector('[aria-controls="hint-drawer"]').setAttribute('aria-expanded', !drawer.hidden);
}

function resetCurrentCode() {
  const ch = CHAPTERS[currentChapterIdx];
  document.getElementById('code-editor').value = ch.lab_code;
  userCodeStore[ch.id] = ch.lab_code;
  updateTodoChips();
  saveLocalState();
}

function fillSolution() {
  const ch = CHAPTERS[currentChapterIdx];
  // 去掉参考答案中的 # >>> SOLUTION 和 # <<< SOLUTION 标记，使代码更整洁
  const cleanRef = ch.ref_code
    .replace(/^\\s*# >>> SOLUTION\\s*\\n/gm, '')
    .replace(/^\\s*# <<< SOLUTION\\s*\\n/gm, '');
  document.getElementById('code-editor').value = cleanRef;
  userCodeStore[ch.id] = cleanRef;
  updateTodoChips();
  saveLocalState();
}

// 内置浏览器端规则与数学等价检验器（支持无后端即开即用评测）
const BROWSER_RULES = {
  "TODO(ch01-1)": { must: [/max\\(/, /exp\\(/, /sum\\(/], msg: "需先减去 logits.max(dim=dim, keepdim=True).values，再求 exp() 并除以 sum(dim=dim, keepdim=True)。" },
  "TODO(ch02-1)": { must: [/ignore_index/, /logsumexp/, /gather/], msg: "需过滤 targets != ignore_index，用 torch.logsumexp 计算对数分母，再用 gather 取出目标 token 的 log_prob 求负均值。" },
  "TODO(ch02-2)": { must: [/exp\\(/], msg: "困惑度 PPL = math.exp(mean_ce_loss)。" },
  "TODO(ch03-1)": { must: [/zip\\(/], msg: "遍历 zip(ids, ids[1:]) 统计相邻字节对出现频次。" },
  "TODO(ch03-2)": { must: [/while\\s+/, /\\+=\\s*2/], msg: "用 while 循环扫描，命中 pair 时追加新 id 并跳过 2 个位置 (i += 2)。" },
  "TODO(ch03-3)": { must: [/get_stats/, /max\\(/, /merge\\(/, /256\\s*\\+/], msg: "循环统计所有 chunk 的 get_stats，选最高频 max(stats, key=stats.get)，分配 256+i 并更新 merges 与 vocab。" },
  "TODO(ch03-4)": { must: [/min\\(/, /merges/, /merge\\(/], msg: "找 merges 中优先级最高（ID 最小）的相邻对 min(stats, key=...) 循环合并。" },
  "TODO(ch04-1)": { must: [/pow\\(2\\)|\\*\\*\\s*2/, /mean\\(-1/, /keepdim\\s*=\\s*True/, /rsqrt/], msg: "需计算 x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)。" },
  "TODO(ch05-1)": { must: [/expand\\(/, /reshape\\(/], msg: "使用 x[:, :, :, None, :].expand(...).reshape(...) 实现零拷贝 KV 头广播。" },
  "TODO(ch05-2)": { must: [/q_proj/, /k_proj/, /v_proj/, /view\\(/, /q_norm/, /k_norm/], msg: "完成 q/k/v 线性投影、view 拆多头，并对 xq, xk 施加 q_norm 和 k_norm。" },
  "TODO(ch05-3)": { must: [/torch\\.cat/, /dim\\s*=\\s*1/, /use_cache/], msg: "当 past_key_value 非空时在 dim=1 拼接历史 K/V，并根据 use_cache 返回 (xk, xv)。" },
  "TODO(ch05-4)": { must: [/transpose\\(-2,\\s*-1\\)/, /sqrt/, /-seq_len:/, /triu\\(1\\)/, /softmax/], msg: "缩放点积 scores = (xq @ xk.transpose(-2,-1)).float() / sqrt(head_dim)，仅对最后 -seq_len: 列加 triu(1) 上三角 -inf 掩码，再 softmax 乘以 xv。" },
  "TODO(ch06-1)": { must: [/rope_base/, /arange\\(0,\\s*dim,\\s*2\\)/], msg: "计算基础频率 1.0 / (rope_base ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))。" },
  "TODO(ch06-2)": { must: [/clamp\\(/, /factor/], msg: "用 torch.clamp 构造 [0,1] 线性斜坡 ramp，并按 freqs * (1 - ramp + ramp / factor) 缩放。" },
  "TODO(ch06-3)": { must: [/outer\\(/, /cos\\(/, /sin\\(/, /cat\\(/], msg: "用 torch.outer(t, freqs) 计算位置角度外积，并将 cos 和 sin 在 dim=-1 复制拼接两份。" },
  "TODO(ch06-4)": { must: [/cat\\(/, /-x\\[/], msg: "将后半部分取负号放到前半部分之前：torch.cat((-x[..., d//2:], x[..., :d//2]), dim=-1)。" },
  "TODO(ch06-5)": { must: [/unsqueeze\\(/, /rotate_half/], msg: "按 q * cos.unsqueeze(unsqueeze_dim) + rotate_half(q) * sin.unsqueeze(unsqueeze_dim) 施加旋转。" },
  "TODO(ch07-1)": { must: [/down_proj/, /act_fn/, /gate_proj/, /up_proj/], msg: "SwiGLU 公式：self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))。" },
  "TODO(ch08-1)": { must: [/input_layernorm/, /self_attn/, /post_attention_layernorm/, /mlp/], msg: "Pre-Norm 残差连接：h = h + self_attn(input_layernorm(h)); h = h + mlp(post_attention_layernorm(h))。" },
  "TODO(ch08-2)": { must: [/freqs_cos\\[start_pos/, /freqs_sin\\[start_pos/], msg: "按 [start_pos : start_pos + seq_length] 从预计算表中切片 cos 与 sin。" },
  "TODO(ch08-3)": { must: [/:-1/, /1:/, /cross_entropy/, /ignore_index\\s*=\\s*-100/], msg: "错位对齐 logits[..., :-1, :] 与 labels[..., 1:] 并计算 F.cross_entropy(..., ignore_index=-100)。" },
  "TODO(ch09-1)": { must: [/topk\\(/, /num_experts_per_tok/], msg: "调用 torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False)。" },
  "TODO(ch09-2)": { must: [/nonzero\\(\\)/, /index_add_/], msg: "筛出命中当前专家的 token_idx，计算 expert(x_flat[token_idx]) * weight 并用 y.index_add_(0, token_idx, ...) 累加。" },
  "TODO(ch09-3)": { must: [/one_hot/, /mean\\(0\\)/, /router_aux_loss_coef/], msg: "计算 load = F.one_hot(topk_idx, num_experts).float().mean(0)，并与 scores.mean(0) 内积乘以 num_experts * coef。" },
  "TODO(ch10-1)": { must: [/unique\\(/, /where\\(/], msg: "找出 input_ids[i] 中出现过的 unique token，用 torch.where(score > 0, score / penalty, score * penalty) 惩罚。" },
  "TODO(ch10-2)": { must: [/topk\\(/, /-float\\(['"]inf['"]\\)/], msg: "将小于 torch.topk(logits, top_k)[0][..., -1, None] 的位置置为 -inf。" },
  "TODO(ch10-3)": { must: [/sort\\(/, /cumsum\\(/, /pad\\(/, /scatter\\(/], msg: "降序 sort -> cumsum(softmax) > top_p -> F.pad(mask[..., :-1], (1, 0)) 右移一位 -> scatter 回原顺序置 -inf。" },
  "TODO(ch10-4)": { must: [/past_len/, /input_ids\\[:,\\s*past_len:\\]/], msg: "增量推理切片：只向模型传入未缓存的新 token input_ids[:, past_len:]。" },
  "TODO(ch11-1)": { must: [/bos_token_id/, /eos_token_id/, /pad_token_id/, /clone\\(\\)/], msg: "前后拼 bos/eos，右侧补 pad_token_id，labels = input_ids.clone() 并将 pad 位置置为 -100。" },
  "TODO(ch13-1)": { must: [/bos_id/, /eos_id/, /start/, /end/], msg: "滑动窗口匹配 bos_id 与 eos_id，将 [start, end + len(eos_id)) 区间的 label 激活。" },
  "TODO(ch12-1)": { must: [/cos\\(/, /math\\.pi/, /0\\.1/, /0\\.45/], msg: "余弦退火公式：lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))。" },
  "TODO(ch12-2)": { must: [/\\/\\s*accumulation_steps/], msg: "梯度累积时需先将单步损失除以 accumulation_steps。" },
  "TODO(ch12-3)": { must: [/accumulation_steps/, /clip_grad_norm_/, /step\\(/, /zero_grad/], msg: "满 accumulation_steps 步时执行 unscale_ -> clip_grad_norm_ -> step -> zero_grad。" },
  "TODO(ch14-1)": { must: [/normal_/, /zero_\\(\\)/], msg: "矩阵 A 用 normal_(0.0, 0.02) 初始化，矩阵 B 用 zero_() 全零初始化。" },
  "TODO(ch14-2)": { must: [/self\\.B\\(self\\.A\\(x\\)\\)/, /scaling/], msg: "前向返回 self.B(self.A(x)) * self.scaling。" },
  "TODO(ch14-3)": { must: [/B\\.weight/, /A\\.weight/, /@/, /scaling/], msg: "权重合并：delta = (module.lora.B.weight.float() @ module.lora.A.weight.float()) * module.lora.scaling.float() 加回原权重。" },
  "TODO(ch15-1)": { must: [/no_grad/, /softmax\\(/, /log_softmax\\(/, /kl_div/, /temperature\\s*\\*\\*\\s*2/], msg: "计算 teacher softmax 与 student log_softmax 的 F.kl_div，并乘以 (temperature ** 2)。" },
  "TODO(ch16-1)": { must: [/log_softmax/, /gather\\(/], msg: "对 logits 求 log_softmax 后用 torch.gather 在词表维取出 labels 对应的对数概率。" },
  "TODO(ch16-2)": { must: [/mask/, /sum\\(dim=1\\)/, /logsigmoid/], msg: "用 mask 对序列 log_probs 求和，切分 chosen/rejected，返回 -F.logsigmoid(beta * logits).mean()。" },
  "TODO(ch17-1)": { must: [/reversed\\(range/, /delta/, /lastgaelam/], msg: "倒序遍历时间步 t，计算 delta = r_t + gamma * V_{t+1} - V_t 与 lastgaelam = delta + gamma * lam * lastgaelam。" },
  "TODO(ch17-2)": { must: [/exp\\(/, /clamp\\(/, /max\\(/], msg: "计算 ratio = torch.exp(logp - old_logp) 与 clamp 后的 PPO-Clip 损失及 clipfrac。" },
  "TODO(ch17-3)": { must: [/clamp\\(/, /0\\.5/, /\\*\\*\\s*2/], msg: "对价值预测同样做 cliprange 截断，取 0.5 * max((V - R)^2, (V_clip - R)^2) 的掩码均值。" },
  "TODO(ch18-1)": { must: [/view\\(-1,\\s*num_generations\\)/, /mean\\(dim=1\\)/, /std\\(dim=1/, /repeat_interleave/], msg: "view 为 [B, G]，求组内 mean 和 std(unbiased=False) 并 repeat_interleave 回 [B*G] 做标准化。" },
  "TODO(ch18-2)": { must: [/cispo/, /clamp\\(ratio,\\s*max=epsilon_high\\)\\.detach\\(\\)/, /min\\(/], msg: "区分 cispo (裁剪重要性采样权重并 detach) 与 grpo (min 裁剪策略比率)，再按序列 mask 求均值。" },
  "TODO(ch19-1)": { must: [/TOOL_CALL_RE\\.findall/, /json\\.loads/], msg: "用正则提取 <tool_call> 内容并 json.loads 解析。" },
  "TODO(ch19-2)": { must: [/count\\(['"]<tool_call>['"]\\)/, /parse_tool_calls/, /validate_gt_in_text/, /2\\.5/], msg: "综合计算标签闭合扣分、合法调用对齐分与 2.5 * GT 命中率分，并 clip 到 [-3, 3]。" }
};

async function submitAndGrade() {
  const ch = CHAPTERS[currentChapterIdx];
  const code = document.getElementById('code-editor').value;
  userCodeStore[ch.id] = code;
  updateTodoChips();

  const statusText = document.getElementById('console-status-text');
  const resultsEl = document.getElementById('testcase-results');
  const request = ++gradeRequest;
  if (editorShare > 94) setEditorShare(Math.min(lastOpenShare, 78));
  resultsEl.textContent = '';
  statusText.textContent = '正在评测…';

  // 若本地 PyTorch 后端在线，优先发往真实 PyTorch 内核执行 pytest
  if (backendAvailable) {
    try {
      const res = await fetch(backendUrl + '/api/grade', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ chapterId: ch.id === 'ch00' ? 'ch01-02' : ch.id, code })
      });
      const data = await res.json();
      if (request !== gradeRequest) return;
      if (data.ok && data.passed) {
        solvedStore[ch.id] = true;
        saveLocalState();
        renderChapterNav();
        statusText.textContent = `✅ PyTorch 真机评测通过 (${data.elapsed_ms} ms)`;
        resultsEl.innerHTML = `<pre class="code-walk-pre">${escapeHtml(data.output)}</pre>`;
        return;
      } else if (data.ok) {
        statusText.textContent = `❌ PyTorch 评测未通过 (${data.elapsed_ms} ms)`;
        resultsEl.innerHTML = `
          <div class="testcase-card testcase-fail"><strong>❌ 单元测试失败（剩余未实现 TODO: ${data.remaining_todos}）</strong></div>
          <pre style="margin-top:8px; font-size:11.5px; color:var(--destructive); white-space:pre-wrap;">${escapeHtml(data.output)}</pre>
        `;
        return;
      }
    } catch (err) {
      backendAvailable = false;
    }
  }

  // 静态网页只检查待实现标记和必需的代码模式。
  if (request !== gradeRequest) return;
  const caseCards = [];
  let allPass = true;
  ch.todos.forEach((todo, i) => {
    const tag = todo.match(/ch\\d+-\\d+/)[0];
    const notImplRe = new RegExp(`raise\\\\s+NotImplementedError\\\\(["']${tag}`);
    if (notImplRe.test(code)) {
      allPass = false;
      caseCards.push(`<div class="testcase-card testcase-fail"><strong>📝 关卡 ${i + 1} (${todo})：尚未实现</strong><br>代码中仍包含 <code>raise NotImplementedError("${tag}...")</code>，请替换为你自己的实现（或点击上方「提示」或「填入参考答案」）。</div>`);
      return;
    }
    const rule = BROWSER_RULES[todo];
    if (rule) {
      const missing = rule.must.filter(re => !re.test(code));
      if (missing.length > 0) {
        allPass = false;
        caseCards.push(`<div class="testcase-card testcase-fail"><strong>❌ 关卡 ${i + 1} (${todo})：核心算子或边界条件缺失</strong><br>💡 诊断提示：${escapeHtml(rule.msg)}</div>`);
        return;
      }
    }
    caseCards.push(`<div class="testcase-card testcase-pass"><strong>✅ 关卡 ${i + 1} (${todo})：已找到练习所需的代码模式。</strong></div>`);
  });

  if (allPass) {
    solvedStore[ch.id] = true;
    saveLocalState();
    renderChapterNav();
    statusText.textContent = '浏览器规则检查通过';
    caseCards.push(`<div class="testcase-card testcase-pass" style="margin-top:8px;"><strong>本章代码结构检查通过。</strong> 在本地运行 PyTorch 评测可进一步验证数值结果。</div>`);
  } else {
    statusText.textContent = '⚠️ 部分关卡待完成或未通过检查';
  }
  resultsEl.innerHTML = caseCards.join('');
}

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('code-editor').addEventListener('input', () => {
    updateTodoChips();
    saveLocalState();
  });
  document.getElementById('code-editor').addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
      heightHandle.focus();
      return;
    }
    if (e.key === 'Tab' && !e.shiftKey) {
      e.preventDefault();
      const start = this.selectionStart;
      const end = this.selectionEnd;
      this.value = this.value.substring(0, start) + '    ' + this.value.substring(end);
      this.selectionStart = this.selectionEnd = start + 4;
      updateTodoChips();
      saveLocalState();
    }
  });
});

// ==================== 5. 八大交互式底层可视化沙盒 ====================
function renderVisualizer(kind) {
  const el = document.getElementById('tab-content-vis');
  if (kind === 'fp_roofline') {
    el.innerHTML = `
      <div class="vis-panel">
        <div style="font-size:14px; font-weight:700; margin-bottom:8px;">🎛️ 实验室 1：IEEE-754 (FP16 vs BF16) 平方溢出检测 & GPU Roofline 算术强度计算器</div>
        <div class="vis-controls">
          <div class="vis-ctrl">
            <label>激活值大小 |x|: <strong id="v-fp-x-lbl">300</strong></label>
            <input type="range" min="10" max="1000" step="10" value="300" id="v-fp-x" oninput="updateFpRoofline()" />
          </div>
          <div class="vis-ctrl">
            <label>输入序列长度 L (Prefill vs Decode=1): <strong id="v-rf-l-lbl">1</strong></label>
            <input type="range" min="1" max="1024" step="1" value="1" id="v-rf-l" oninput="updateFpRoofline()" />
          </div>
          <div class="vis-ctrl">
            <label>Batch Size B: <strong id="v-rf-b-lbl">1</strong></label>
            <input type="range" min="1" max="32" step="1" value="1" id="v-rf-b" oninput="updateFpRoofline()" />
          </div>
        </div>
        <div class="metric-grid" id="v-fp-metrics"></div>
        <div id="v-fp-explain" style="font-size:12.5px; margin-top:8px;"></div>
      </div>
    `;
    updateFpRoofline();
  } else if (kind === 'softmax_ce') {
    el.innerHTML = `
      <div class="vis-panel">
        <div style="font-size:14px; font-weight:700; margin-bottom:8px;">🎛️ 实验室 2：Softmax 指数上溢防护、LogSumExp 交叉熵与梯度 (p - y) 模拟器</div>
        <div class="vis-controls">
          <div class="vis-ctrl">
            <label>Logits 整体放大倍数 (测试大数值溢出): <strong id="v-sm-scale-lbl">1.0</strong></label>
            <input type="range" min="1" max="120" step="1" value="1" id="v-sm-scale" oninput="updateSoftmaxVis()" />
          </div>
          <div class="vis-ctrl">
            <label>正确目标 Token (y=0) 的额外加分: <strong id="v-sm-boost-lbl">+1.5</strong></label>
            <input type="range" min="-4" max="8" step="0.5" value="1.5" id="v-sm-boost" oninput="updateSoftmaxVis()" />
          </div>
        </div>
        <div class="metric-grid" id="v-sm-metrics"></div>
        <div id="v-sm-bars"></div>
      </div>
    `;
    updateSoftmaxVis();
  } else if (kind === 'bpe_vis') {
    el.innerHTML = `
      <div class="vis-panel">
        <div style="font-size:14px; font-weight:700; margin-bottom:8px;">🎛️ 实验室 3：Byte-Level BPE 迭代合并 & 中文 3 字节 UTF-8 流式解码缓冲演示</div>
        <div class="vis-controls">
          <div class="vis-ctrl">
            <label>BPE 合并轮数 Step: <strong id="v-bpe-step-lbl">0 / 4</strong></label>
            <input type="range" min="0" max="4" step="1" value="0" id="v-bpe-step" oninput="updateBpeVis()" />
          </div>
        </div>
        <div id="v-bpe-body"></div>
      </div>
    `;
    updateBpeVis();
  } else if (kind === 'norm_grad') {
    el.innerHTML = `
      <div class="vis-panel">
        <div style="font-size:14px; font-weight:700; margin-bottom:8px;">🎛️ 实验室 4：Pre-Norm vs Post-Norm 深层 Transformer 各层反向梯度范数对比</div>
        <div class="vis-controls">
          <div class="vis-ctrl">
            <label>网络总层数 L: <strong id="v-ng-l-lbl">16 层</strong></label>
            <input type="range" min="4" max="32" step="2" value="16" id="v-ng-l" oninput="updateNormGradVis()" />
          </div>
        </div>
        <div id="v-ng-bars"></div>
      </div>
    `;
    updateNormGradVis();
  } else if (kind === 'gqa_kv') {
    el.innerHTML = `
      <div class="vis-panel">
        <div style="font-size:14px; font-weight:700; margin-bottom:8px;">🎛️ 实验室 5：GQA 头分组共享、KV Cache 显存计算器与非方阵因果掩码 [L_cur, L_past + L_cur]</div>
        <div class="vis-controls">
          <div class="vis-ctrl">
            <label>KV 头数 H_kv (H_q 固定为 8): <strong id="v-gqa-kv-lbl">4 (GQA 2:1)</strong></label>
            <input type="range" min="1" max="8" step="1" value="4" id="v-gqa-kv" oninput="updateGqaVis()" />
          </div>
          <div class="vis-ctrl">
            <label>已缓存历史长度 L_past: <strong id="v-gqa-past-lbl">3</strong></label>
            <input type="range" min="0" max="6" step="1" value="3" id="v-gqa-past" oninput="updateGqaVis()" />
          </div>
          <div class="vis-ctrl">
            <label>当前输入长度 seq_len: <strong id="v-gqa-cur-lbl">3</strong></label>
            <input type="range" min="1" max="5" step="1" value="3" id="v-gqa-cur" oninput="updateGqaVis()" />
          </div>
        </div>
        <div class="metric-grid" id="v-gqa-metrics"></div>
        <div id="v-gqa-matrix"></div>
      </div>
    `;
    updateGqaVis();
  } else if (kind === 'rope_yarn') {
    el.innerHTML = `
      <div class="vis-panel">
        <div style="font-size:14px; font-weight:700; margin-bottom:8px;">🎛️ 实验室 6：RoPE 复平面旋转相对距离不变性验证 & YaRN 高低频分治斜坡</div>
        <div class="vis-controls">
          <div class="vis-ctrl">
            <label>Query 位置 m: <strong id="v-rope-m-lbl">7</strong></label>
            <input type="range" min="0" max="20" step="1" value="7" id="v-rope-m" oninput="updateRopeVis()" />
          </div>
          <div class="vis-ctrl">
            <label>Key 位置 n: <strong id="v-rope-n-lbl">3</strong></label>
            <input type="range" min="0" max="20" step="1" value="3" id="v-rope-n" oninput="updateRopeVis()" />
          </div>
          <div class="vis-ctrl">
            <label>整体平移偏移量 +Δ (验证内积不变): <strong id="v-rope-shift-lbl">+0</strong></label>
            <input type="range" min="0" max="50" step="5" value="0" id="v-rope-shift" oninput="updateRopeVis()" />
          </div>
        </div>
        <div class="metric-grid" id="v-rope-metrics"></div>
        <div id="v-rope-svg" style="display:flex; justify-content:center; margin-top:8px;"></div>
      </div>
    `;
    updateRopeVis();
  } else if (kind === 'param_calc' || kind === 'moe_vis') {
    el.innerHTML = `
      <div class="vis-panel">
        <div style="font-size:14px; font-weight:700; margin-bottom:8px;">🎛️ 实验室 7：MiniMind Dense / MoE 参数精算器与 Switch 负载均衡损失模拟器</div>
        <div class="vis-controls">
          <div class="vis-ctrl">
            <label>隐藏维 d_model: <strong id="v-pm-d-lbl">768</strong></label>
            <input type="range" min="256" max="1024" step="128" value="768" id="v-pm-d" oninput="updateParamMoEVis()" />
          </div>
          <div class="vis-ctrl">
            <label>层数 n_layers: <strong id="v-pm-l-lbl">8</strong></label>
            <input type="range" min="2" max="16" step="2" value="8" id="v-pm-l" oninput="updateParamMoEVis()" />
          </div>
          <div class="vis-ctrl">
            <label>MoE 专家数 E (1=Dense): <strong id="v-pm-e-lbl">4</strong></label>
            <input type="range" min="1" max="8" step="1" value="4" id="v-pm-e" oninput="updateParamMoEVis()" />
          </div>
        </div>
        <div class="metric-grid" id="v-pm-metrics"></div>
        <div id="v-pm-bars"></div>
      </div>
    `;
    updateParamMoEVis();
  } else if (kind === 'sampling_vis') {
    el.innerHTML = `
      <div class="vis-panel">
        <div style="font-size:14px; font-weight:700; margin-bottom:8px;">🎛️ 实验室 8：Temperature / Repetition Penalty / Top-K / Top-P 核采样实时沙盒</div>
        <div class="vis-controls">
          <div class="vis-ctrl">
            <label>温度 Temperature τ: <strong id="v-sp-t-lbl">0.85</strong></label>
            <input type="range" min="0.2" max="2.0" step="0.05" value="0.85" id="v-sp-t" oninput="updateSamplingVis()" />
          </div>
          <div class="vis-ctrl">
            <label>Top-K: <strong id="v-sp-k-lbl">4</strong></label>
            <input type="range" min="1" max="6" step="1" value="4" id="v-sp-k" oninput="updateSamplingVis()" />
          </div>
          <div class="vis-ctrl">
            <label>Top-P (核采样阈值): <strong id="v-sp-p-lbl">0.85</strong></label>
            <input type="range" min="0.3" max="1.0" step="0.05" value="0.85" id="v-sp-p" oninput="updateSamplingVis()" />
          </div>
        </div>
        <div id="v-sp-bars"></div>
      </div>
    `;
    updateSamplingVis();
  } else if (kind === 'mask_vis') {
    el.innerHTML = `
      <div class="vis-panel">
        <div style="font-size:14px; font-weight:700; margin-bottom:8px;">🎛️ 实验室 9：Pretrain vs SFT vs 多轮 Tool-Call Agent 的 Loss Mask (-100) 透视镜</div>
        <div style="display:flex; gap:8px; margin-bottom:10px;">
          <button class="btn btn-primary" onclick="updateMaskVis('pretrain')" id="btn-mv-pt">1. 预训练 (Pretrain)</button>
          <button class="btn" onclick="updateMaskVis('sft')" id="btn-mv-sft">2. 多轮指令微调 (SFT)</button>
          <button class="btn" onclick="updateMaskVis('agent')" id="btn-mv-ag">3. 多轮工具调用 (Agentic RL)</button>
        </div>
        <div id="v-mv-tokens"></div>
      </div>
    `;
    updateMaskVis('sft');
  } else if (kind === 'lora_kd') {
    el.innerHTML = `
      <div class="vis-panel">
        <div style="font-size:14px; font-weight:700; margin-bottom:8px;">🎛️ 实验室 10：LoRA 低秩参数压缩比 & 知识蒸馏温度 T² 软标签暗知识放大器</div>
        <div class="vis-controls">
          <div class="vis-ctrl">
            <label>LoRA 秩 Rank r (d=768): <strong id="v-lk-r-lbl">16</strong></label>
            <input type="range" min="2" max="64" step="2" value="16" id="v-lk-r" oninput="updateLoraKdVis()" />
          </div>
          <div class="vis-ctrl">
            <label>蒸馏温度 Temperature T: <strong id="v-lk-t-lbl">2.0</strong></label>
            <input type="range" min="0.5" max="5.0" step="0.5" value="2.0" id="v-lk-t" oninput="updateLoraKdVis()" />
          </div>
        </div>
        <div class="metric-grid" id="v-lk-metrics"></div>
        <div id="v-lk-bars"></div>
      </div>
    `;
    updateLoraKdVis();
  } else {
    el.innerHTML = `
      <div class="vis-panel">
        <div style="font-size:14px; font-weight:700; margin-bottom:8px;">🎛️ 实验室 11：PPO/GRPO Token-Clip vs MiniMax CISPO 裁剪权重损失与梯度对比</div>
        <div class="vis-controls">
          <div class="vis-ctrl">
            <label>重要性采样概率比 r_t = π_θ / π_old: <strong id="v-rl-r-lbl">1.60</strong></label>
            <input type="range" min="0.2" max="3.0" step="0.05" value="1.60" id="v-rl-r" oninput="updateRlVis()" />
          </div>
          <div class="vis-ctrl">
            <label>组内相对优势 A_t: <strong id="v-rl-a-lbl">+1.50</strong></label>
            <input type="range" min="-2.0" max="2.0" step="0.25" value="1.50" id="v-rl-a" oninput="updateRlVis()" />
          </div>
        </div>
        <div class="metric-grid" id="v-rl-metrics"></div>
        <div id="v-rl-explain" style="font-size:12.5px; margin-top:8px;"></div>
      </div>
    `;
    updateRlVis();
  }
}

function updateFpRoofline() {
  const x = parseFloat(document.getElementById('v-fp-x').value);
  const L = parseInt(document.getElementById('v-rf-l').value);
  const B = parseInt(document.getElementById('v-rf-b').value);
  document.getElementById('v-fp-x-lbl').textContent = x;
  document.getElementById('v-rf-l-lbl').textContent = L === 1 ? '1 (Decode 逐词生成)' : L + ' (Prefill 批量预填充)';
  document.getElementById('v-rf-b-lbl').textContent = B;

  const xSq = x * x;
  const fp16Overflow = xSq > 65504;
  const intensity = (B * L).toFixed(1);
  const boundType = (B * L < 208) ? '带宽可能先达到上限' : '算力可能先达到上限';

  document.getElementById('v-fp-metrics').innerHTML = `
    <div class="metric-card"><div class="metric-label">x² 真实数值</div><div class="metric-val">${xSq.toLocaleString()}</div></div>
    <div class="metric-card"><div class="metric-label">FP16 直接算 x² (上限 65504)</div><div class="metric-val" style="color:${fp16Overflow ? 'var(--destructive)' : 'var(--success)'}">${fp16Overflow ? '+∞（上溢）' : xSq}</div></div>
    <div class="metric-card"><div class="metric-label">MiniMind 用 FP32 平方</div><div class="metric-val" style="color:var(--success)">${xSq.toLocaleString()}</div></div>
    <div class="metric-card"><div class="metric-label">线性层算术强度 (FLOPs/Byte)</div><div class="metric-val">≈ ${intensity}</div></div>
  `;
  document.getElementById('v-fp-explain').innerHTML = `
    按每个 BF16 权重读一次、忽略激活读写的估算，每字节约完成 ${intensity} 次浮点运算。
    与示例设备约 208 FLOP/byte 的拐点相比，${boundType}。
  `;
}

function updateSoftmaxVis() {
  const scale = parseFloat(document.getElementById('v-sm-scale').value);
  const boost = parseFloat(document.getElementById('v-sm-boost').value);
  document.getElementById('v-sm-scale-lbl').textContent = scale.toFixed(1) + 'x';
  document.getElementById('v-sm-boost-lbl').textContent = (boost >= 0 ? '+' : '') + boost.toFixed(1);

  const base = [1.2 + boost, 0.8, 0.3, -0.5, -1.1];
  const names = ['Token 0 (真实目标 y=0)', 'Token 1', 'Token 2', 'Token 3', 'Token 4'];
  const z = base.map(v => v * scale);
  const naiveOverflow = z.some(v => v > 88); // FP32 exp overflow around 88.7
  const M = Math.max(...z);
  const exps = z.map(v => Math.exp(v - M));
  const sumExp = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map(e => e / sumExp);
  const ce = -(z[0] - (M + Math.log(sumExp)));
  const ppl = Math.exp(ce);

  document.getElementById('v-sm-metrics').innerHTML = `
    <div class="metric-card"><div class="metric-label">朴素 exp(z) 是否溢出</div><div class="metric-val" style="color:${naiveOverflow ? 'var(--destructive)' : 'var(--success)'}">${naiveOverflow ? 'Overflow (NaN) 💥' : '未溢出'}</div></div>
    <div class="metric-card"><div class="metric-label">稳定 Softmax p_y</div><div class="metric-val" style="color:var(--success)">${(probs[0] * 100).toFixed(2)}%</div></div>
    <div class="metric-card"><div class="metric-label">交叉熵 Loss (-log p_y)</div><div class="metric-val">${ce.toFixed(4)}</div></div>
    <div class="metric-card"><div class="metric-label">困惑度 PPL = exp(Loss)</div><div class="metric-val">${ppl.toFixed(2)}</div></div>
  `;
  document.getElementById('v-sm-bars').innerHTML = probs.map((p, i) => {
    const grad = (p - (i === 0 ? 1 : 0)).toFixed(3);
    return `
      <div class="bar-row">
        <span style="width:170px;">${names[i]}</span>
        <div class="bar-track"><div class="bar-fill" style="width:${(p * 100).toFixed(1)}%; background:${i === 0 ? 'var(--success)' : 'var(--chart-1)'}"></div></div>
        <span style="width:165px; text-align:right;">p=${(p * 100).toFixed(1)}% | ∂L/∂z=${grad}</span>
      </div>
    `;
  }).join('');
}

function updateBpeVis() {
  const step = parseInt(document.getElementById('v-bpe-step').value);
  document.getElementById('v-bpe-step-lbl').textContent = step + ' / 4';
  const stages = [
    { rule: '初始状态 (UTF-8 单字节序列)', tokens: ['l', 'o', 'w', ' ', 'l', 'o', 'w', 'e', 'r', ' ', 'l', 'o', 'w', 'e', 's', 't'], note: '共 16 个单字节 token' },
    { rule: '第 1 轮合并: ("l", "o") -> "lo" (最高频出现 3 次, 新 ID=256)', tokens: ['lo', 'w', ' ', 'lo', 'w', 'e', 'r', ' ', 'lo', 'w', 'e', 's', 't'], note: 'token 数从 16 降至 13' },
    { rule: '第 2 轮合并: ("lo", "w") -> "low" (出现 3 次, 新 ID=257)', tokens: ['low', ' ', 'low', 'e', 'r', ' ', 'low', 'e', 's', 't'], note: 'token 数从 13 降至 10，成功学出词根 "low"！' },
    { rule: '第 3 轮合并: (" ", "low") -> " low" (出现 2 次, 新 ID=258)', tokens: ['low', ' low', 'e', 'r', ' low', 'e', 's', 't'], note: 'token 数降至 8（Byte-Level BPE 会将前导空格与词根合并）' },
    { rule: '第 4 轮合并: (" low", "e") -> " lowe" (出现 2 次, 新 ID=259)', tokens: ['low', ' lowe', 'r', ' lowe', 's', 't'], note: 'token 数降至 6，压缩率达到 16 / 6 = 2.67x' }
  ];
  const cur = stages[step];
  document.getElementById('v-bpe-body').innerHTML = `
    <div style="font-size:13px; font-weight:700; color:var(--primary); margin-bottom:6px;">${cur.rule}（${cur.note}）</div>
    <div>${cur.tokens.map(t => `<span class="token-badge token-active">${escapeHtml(JSON.stringify(t))}</span>`).join('')}</div>
    <div style="margin-top:10px; font-size:12px; color:var(--muted-foreground);">
      💡 <strong>中文 UTF-8 流式缓冲示例</strong>：汉字 <code>"智"</code> 的 UTF-8 字节为 <code>[0xE6, 0x99, 0xBA]</code>。若 Token A = <code>[0xE6, 0x99]</code>，单解 Token A 会产生 <code>"\\ufffd"</code>，<code>train_tokenizer.py</code> 检测到 <code>'\\ufffd' in current_decode</code> 时暂存 <code>token_cache</code>，待 Token B = <code>[0xBA]</code> 到达合并为 <code>[0xE6, 0x99, 0xBA]</code> 后一次性解码出完整的 <code>"智"</code>！
    </div>
  `;
}

function updateNormGradVis() {
  const L = parseInt(document.getElementById('v-ng-l').value);
  document.getElementById('v-ng-l-lbl').textContent = L + ' 层';
  const rows = [];
  for (let l = 1; l <= L; l += Math.max(1, Math.floor(L / 8))) {
    const preGrad = 1.0;
    const postGrad = Math.pow(0.75, L - l);
    rows.push(`
      <div class="bar-row">
        <span style="width:70px;">第 ${l} 层</span>
        <div class="bar-track" title="Pre-Norm"><div class="bar-fill" style="width:${preGrad * 95}%; background:var(--success)"></div></div>
        <div class="bar-track" title="Post-Norm"><div class="bar-fill" style="width:${Math.max(2, postGrad * 95)}%; background:var(--destructive)"></div></div>
        <span style="width:175px; text-align:right;">Pre: 1.00 | Post: ${postGrad.toFixed(3)}</span>
      </div>
    `);
  }
  document.getElementById('v-ng-bars').innerHTML = `
    <div style="font-size:12px; margin-bottom:6px;"><span style="color:var(--success)">■ 绿色：Pre-Norm (恒等残差高速公路，浅层梯度不衰减)</span> ｜ <span style="color:var(--destructive)">■ 红色：Post-Norm (浅层梯度随深度指数消失)</span></div>
    ${rows.join('')}
  `;
}

function updateGqaVis() {
  const kv = parseInt(document.getElementById('v-gqa-kv').value);
  const past = parseInt(document.getElementById('v-gqa-past').value);
  const cur = parseInt(document.getElementById('v-gqa-cur').value);
  const validKv = [1, 2, 4, 8].reduce((prev, c) => Math.abs(c - kv) < Math.abs(prev - kv) ? c : prev, 4);
  const nRep = 8 / validKv;
  const label = validKv === 8 ? '8 (标准 MHA 1:1)' : (validKv === 1 ? '1 (极端 MQA 8:1)' : `${validKv} (GQA ${nRep}:1)`);
  document.getElementById('v-gqa-kv-lbl').textContent = label;
  document.getElementById('v-gqa-past-lbl').textContent = past;
  document.getElementById('v-gqa-cur-lbl').textContent = cur;

  const totalLen = past + cur;
  const memRatio = ((validKv / 8) * 100).toFixed(0);
  document.getElementById('v-gqa-metrics').innerHTML = `
    <div class="metric-card"><div class="metric-label">repeat_kv 倍数 n_rep</div><div class="metric-val">${nRep}x</div></div>
    <div class="metric-card"><div class="metric-label">KV Cache 显存占 MHA 比例</div><div class="metric-val" style="color:var(--success)">${memRatio}%</div></div>
    <div class="metric-card"><div class="metric-label">scores 矩阵形状</div><div class="metric-val">[${cur}, ${totalLen}]</div></div>
  `;

  let table = `<div style="font-size:12px; margin-top:8px;"><strong>非方阵因果掩码可视化 <code>scores[:, :, :, -seq_len:] += triu(1)</code>（行=当前 Query，列=历史 Key + 当前 Key）：</strong></div><div style="overflow-x:auto; margin-top:6px;"><table style="border-collapse:collapse; font-family:var(--font-mono); font-size:11.5px;">`;
  for (let r = 0; r < cur; r++) {
    table += `<tr><td style="padding:4px 8px;">Q_cur[${r}] (pos=${past + r})</td>`;
    for (let c = 0; c < totalLen; c++) {
      const isPast = c < past;
      const masked = !isPast && (c - past > r);
      const bg = masked ? 'var(--destructive-surface)' : (isPast ? 'var(--accent-surface)' : 'rgba(16,185,129,0.18)');
      const txt = masked ? '-inf (遮蔽)' : (isPast ? `K_past[${c}] ✅` : `K_cur[${c - past}] ✅`);
      table += `<td style="border:1px solid var(--border); padding:4px 8px; background:${bg};">${txt}</td>`;
    }
    table += `</tr>`;
  }
  table += `</table></div>`;
  document.getElementById('v-gqa-matrix').innerHTML = table;
}

function updateRopeVis() {
  const m = parseInt(document.getElementById('v-rope-m').value);
  const n = parseInt(document.getElementById('v-rope-n').value);
  const shift = parseInt(document.getElementById('v-rope-shift').value);
  document.getElementById('v-rope-m-lbl').textContent = m;
  document.getElementById('v-rope-n-lbl').textContent = n;
  document.getElementById('v-rope-shift-lbl').textContent = '+' + shift;

  const theta = 0.35;
  const angM = (m + shift) * theta;
  const angN = (n + shift) * theta;
  const dot = Math.cos(angM - angN);

  document.getElementById('v-rope-metrics').innerHTML = `
    <div class="metric-card"><div class="metric-label">平移后位置 (m+Δ, n+Δ)</div><div class="metric-val">(${m + shift}, ${n + shift})</div></div>
    <div class="metric-card"><div class="metric-label">相对距离 (m+Δ) - (n+Δ)</div><div class="metric-val">${m - n} (恒定不变!)</div></div>
    <div class="metric-card"><div class="metric-label">旋转后内积 &lt;R_m q, R_n k&gt;</div><div class="metric-val" style="color:var(--success)">${dot.toFixed(4)}</div></div>
  `;

  const cx = 90, cy = 90, r = 68;
  const qx = cx + r * Math.cos(angM), qy = cy - r * Math.sin(angM);
  const kx = cx + r * Math.cos(angN), ky = cy - r * Math.sin(angN);
  document.getElementById('v-rope-svg').innerHTML = `
    <svg width="200" height="180" style="background:var(--card); border:1px solid var(--border); border-radius:8px;">
      <circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="var(--border)" stroke-width="1.5" />
      <line x1="${cx}" y1="${cy}" x2="${qx}" y2="${qy}" stroke="var(--chart-1)" stroke-width="3" />
      <line x1="${cx}" y1="${cy}" x2="${kx}" y2="${ky}" stroke="var(--success)" stroke-width="3" />
      <circle cx="${qx}" cy="${qy}" r="4" fill="var(--chart-1)" />
      <circle cx="${kx}" cy="${ky}" r="4" fill="var(--success)" />
      <text x="10" y="18" fill="var(--chart-1)" font-size="11">■ R_m q (位置 ${m+shift})</text>
      <text x="10" y="34" fill="var(--success)" font-size="11">■ R_n k (位置 ${n+shift})</text>
    </svg>
  `;
}

function updateParamMoEVis() {
  const d = parseInt(document.getElementById('v-pm-d').value);
  const L = parseInt(document.getElementById('v-pm-l').value);
  const E = parseInt(document.getElementById('v-pm-e').value);
  document.getElementById('v-pm-d-lbl').textContent = d;
  document.getElementById('v-pm-l-lbl').textContent = L;
  document.getElementById('v-pm-e-lbl').textContent = E === 1 ? '1 (Dense)' : `${E} Experts (Top-1 MoE)`;

  const V = 6400;
  const I = Math.ceil((d * Math.PI) / 64) * 64;
  const embedParams = V * d;
  const attnPerLayer = 2 * d * d + 2 * d * (d / 2) + 2 * (d / 8);
  const ffnSingle = 3 * d * I;
  const activeTotal = (embedParams + L * (attnPerLayer + ffnSingle + 2 * d) + d) / 1e6;
  const totalParams = (embedParams + L * (attnPerLayer + E * ffnSingle + (E > 1 ? d * E : 0) + 2 * d) + d) / 1e6;

  document.getElementById('v-pm-metrics').innerHTML = `
    <div class="metric-card"><div class="metric-label">SwiGLU 中间维 I (对齐64)</div><div class="metric-val">${I}</div></div>
    <div class="metric-card"><div class="metric-label">总参数量 Total Params</div><div class="metric-val">${totalParams.toFixed(2)} M</div></div>
    <div class="metric-card"><div class="metric-label">单 Token 激活参数 Active</div><div class="metric-val" style="color:var(--success)">${activeTotal.toFixed(2)} M</div></div>
  `;
  const embedPct = ((embedParams / 1e6) / totalParams * 100).toFixed(1);
  const attnPct = (((L * attnPerLayer) / 1e6) / totalParams * 100).toFixed(1);
  const ffnPct = (((L * E * ffnSingle) / 1e6) / totalParams * 100).toFixed(1);
  document.getElementById('v-pm-bars').innerHTML = `
    <div class="bar-row"><span style="width:140px;">共享 Embedding</span><div class="bar-track"><div class="bar-fill" style="width:${embedPct}%; background:var(--chart-3)"></div></div><span style="width:90px; text-align:right;">${embedPct}%</span></div>
    <div class="bar-row"><span style="width:140px;">GQA Attention</span><div class="bar-track"><div class="bar-fill" style="width:${attnPct}%; background:var(--chart-1)"></div></div><span style="width:90px; text-align:right;">${attnPct}%</span></div>
    <div class="bar-row"><span style="width:140px;">SwiGLU / MoE FFN</span><div class="bar-track"><div class="bar-fill" style="width:${ffnPct}%; background:var(--success)"></div></div><span style="width:90px; text-align:right;">${ffnPct}%</span></div>
  `;
}

function updateSamplingVis() {
  const T = parseFloat(document.getElementById('v-sp-t').value);
  const K = parseInt(document.getElementById('v-sp-k').value);
  const P = parseFloat(document.getElementById('v-sp-p').value);
  document.getElementById('v-sp-t-lbl').textContent = T.toFixed(2);
  document.getElementById('v-sp-k-lbl').textContent = K;
  document.getElementById('v-sp-p-lbl').textContent = P.toFixed(2);

  const raw = [
    { w: '"人工智能"', z: 3.2 },
    { w: '"大模型"', z: 2.6 },
    { w: '"深度学习"', z: 1.9 },
    { w: '"机器学习"', z: 1.2 },
    { w: '"神经网络"', z: 0.4 },
    { w: '"乱码词"', z: -0.8 }
  ];
  const scaled = raw.map((item, idx) => ({ ...item, zT: idx < K ? item.z / T : -Infinity }));
  const maxZ = Math.max(...scaled.map(s => s.zT));
  const exps = scaled.map(s => s.zT === -Infinity ? 0 : Math.exp(s.zT - maxZ));
  const sumE = exps.reduce((a, b) => a + b, 0);
  let cum = 0;
  const finalItems = scaled.map((s, i) => {
    const prob = exps[i] / sumE;
    const prevCum = cum;
    cum += prob;
    const kept = (i < K) && (prevCum < P); // 右移一位掩码：prevCum < P 的词均被保留
    return { ...s, prob, kept };
  });
  const keptSum = finalItems.reduce((a, b) => a + (b.kept ? b.prob : 0), 0);

  document.getElementById('v-sp-bars').innerHTML = finalItems.map(item => {
    const normP = item.kept ? (item.prob / keptSum * 100) : 0;
    return `
      <div class="bar-row">
        <span style="width:110px;">${item.w}</span>
        <div class="bar-track"><div class="bar-fill" style="width:${normP.toFixed(1)}%; background:${item.kept ? 'var(--success)' : 'var(--destructive)'}"></div></div>
        <span style="width:165px; text-align:right;">${item.kept ? '保留 ' + normP.toFixed(1) + '%' : '被 Top-K/Top-P 过滤 (-inf)'}</span>
      </div>
    `;
  }).join('');
}

function updateMaskVis(mode) {
  ['pt', 'sft', 'ag'].forEach(k => document.getElementById('btn-mv-' + k).className = 'btn');
  document.getElementById(mode === 'pretrain' ? 'btn-mv-pt' : (mode === 'sft' ? 'btn-mv-sft' : 'btn-mv-ag')).className = 'btn btn-primary';

  let items;
  if (mode === 'pretrain' || mode === 'sft') {
    // Compare the supervision on exactly the same tokenized text. The first token has no predecessor.
    items = [
      ['<|im_start|>', false], ['user\\n', false], ['1+1 等于几？', false], ['<|im_end|>\\n', false],
      ['<|im_start|>assistant\\n', false], ['等于 2。', true], ['<|im_end|>\\n', true], ['<|pad|>', false]
    ];
    if (mode === 'pretrain') items = items.map(([text], i) => [text, i > 0 && i < items.length - 1]);
  } else {
    items = [
      ['<|im_start|>user\\n算 256*37<|im_end|>\\n', false],
      ['<|im_start|>assistant\\n', false],
      ['<tool_call>{"name":"calculate_math","arguments":{"expression":"256*37"}}</tool_call>\\n', true],
      ['<|im_end|>\\n<|im_start|>user\\n<tool_response>{"result":"9472"}</tool_response><|im_end|>\\n', false],
      ['<|im_start|>assistant\\n', false],
      ['计算结果是 9472。', true], ['<|im_end|>', false]
    ];
  }
  document.getElementById('v-mv-tokens').innerHTML = `
    <div style="font-size:12px; margin-bottom:8px;"><span style="color:var(--success)">■ 绿色 · 计入该位置的训练目标</span> ｜ <span style="color:var(--muted-foreground)">■ 灰色 · 跳过该位置的直接损失</span></div>
    <div>${items.map(([txt, active]) => `<span class="token-badge ${active ? 'token-active' : 'token-masked'}">${escapeHtml(txt)}</span>`).join('')}</div>
  `;
}

function updateLoraKdVis() {
  const r = parseInt(document.getElementById('v-lk-r').value);
  const T = parseFloat(document.getElementById('v-lk-t').value);
  document.getElementById('v-lk-r-lbl').textContent = r;
  document.getElementById('v-lk-t-lbl').textContent = T.toFixed(1);

  const d = 768;
  const fullParams = d * d;
  const loraParams = r * (d + d);
  const ratio = (loraParams / fullParams * 100).toFixed(2);

  const tLogits = [4.0, 2.2, 1.0, -1.5];
  const names = ['最佳词 "窗台"', '近义词 "桌面" (暗知识!)', '近义词 "草地" (暗知识!)', '无关词 "钢筋"'];
  const exps = tLogits.map(z => Math.exp(z / T));
  const sumE = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map(e => e / sumE);

  document.getElementById('v-lk-metrics').innerHTML = `
    <div class="metric-card"><div class="metric-label">原方阵 W 参数 (768×768)</div><div class="metric-val">${fullParams.toLocaleString()}</div></div>
    <div class="metric-card"><div class="metric-label">LoRA (A+B) 参数量</div><div class="metric-val" style="color:var(--success)">${loraParams.toLocaleString()} (${ratio}%)</div></div>
    <div class="metric-card"><div class="metric-label">KL 梯度补偿乘子 T²</div><div class="metric-val">×${(T * T).toFixed(2)}</div></div>
  `;
  document.getElementById('v-lk-bars').innerHTML = probs.map((p, i) => `
    <div class="bar-row">
      <span style="width:170px;">${names[i]}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${(p * 100).toFixed(1)}%; background:${i === 0 ? 'var(--chart-1)' : 'var(--success)'}"></div></div>
      <span style="width:90px; text-align:right;">${(p * 100).toFixed(1)}%</span>
    </div>
  `).join('');
}

function updateRlVis() {
  const r = parseFloat(document.getElementById('v-rl-r').value);
  const A = parseFloat(document.getElementById('v-rl-a').value);
  document.getElementById('v-rl-r-lbl').textContent = r.toFixed(2);
  document.getElementById('v-rl-a-lbl').textContent = (A >= 0 ? '+' : '') + A.toFixed(2);

  const eps = 0.2;
  const epsHigh = 5.0;
  const clippedR = Math.min(Math.max(r, 1 - eps), 1 + eps);
  const ppoClipTriggered = (A > 0 && r > 1 + eps) || (A < 0 && r < 1 - eps);
  const ppoGrad = ppoClipTriggered ? 0 : -r * A;
  const cispoWeight = Math.min(r, epsHigh);
  const cispoGrad = -cispoWeight * A;

  document.getElementById('v-rl-metrics').innerHTML = `
    <div class="metric-card"><div class="metric-label">PPO / GRPO 是否触发裁剪</div><div class="metric-val" style="color:${ppoClipTriggered ? 'var(--warning)' : 'var(--success)'}">${ppoClipTriggered ? '已触发 (梯度归零!)' : '信任域内 (正常更新)'}</div></div>
    <div class="metric-card"><div class="metric-label">PPO/GRPO 对 log π 的梯度系数</div><div class="metric-val">${ppoGrad.toFixed(2)}</div></div>
    <div class="metric-card"><div class="metric-label">CISPO 对 log π 的梯度系数</div><div class="metric-val" style="color:var(--success)">${cispoGrad.toFixed(2)} (保留梯度!)</div></div>
  `;
  document.getElementById('v-rl-explain').innerHTML = ppoClipTriggered
    ? `⚠️ 当前概率比 r_t=${r.toFixed(2)} 超出了 [0.8, 1.2] 区间：传统 PPO/GRPO 直接将该 token 的梯度砍成 <strong>0</strong>；而 MiniMind 默认采用的 <strong>CISPO</strong> 仅对权重封顶 (≤5.0)，依然保留 <strong>${cispoGrad.toFixed(2)}</strong> 的有效梯度更新！`
    : `✅ 当前概率比 r_t=${r.toFixed(2)} 处于 [0.8, 1.2] 信任域内，PPO/GRPO 与 CISPO 梯度方向与大小完全一致。`;
}

// ==================== 6. 多设备持久化与云同步 (localStorage / URL Sync / JSON / GitHub Gist) ====================
const STORAGE_KEY = 'learn_minimind_state_v1';
const GIST_CFG_KEY = 'learn_minimind_gist_cfg_v1';

function setSyncStatus(msg, isError) {
  const el = document.getElementById('sync-status-msg');
  if (el) {
    el.textContent = msg;
    el.style.color = isError ? 'var(--destructive)' : 'var(--success)';
  }
}

function saveLocalState() {
  try {
    const curCh = CHAPTERS[currentChapterIdx];
    const editor = document.getElementById('code-editor');
    if (displayedChapterId && curCh && editor) {
      userCodeStore[curCh.id] = editor.value;
    }
    // 仅存储与初始 lab_code 不同的自定义代码，大幅减小存储体积
    const customCode = {};
    CHAPTERS.forEach(ch => {
      if (typeof userCodeStore[ch.id] === 'string' && userCodeStore[ch.id] !== ch.lab_code) {
        customCode[ch.id] = userCodeStore[ch.id];
      }
    });
    const payload = {
      v: 1,
      ts: new Date().toISOString(),
      chapterIdx: currentChapterIdx,
      theme: document.documentElement.getAttribute('data-theme') || 'dark',
      solved: solvedStore,
      code: customCode
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
  } catch (e) {}
}

function applyStateSnapshot(snap, silent) {
  if (!snap || typeof snap !== 'object') return false;
  if (displayedChapterId) userCodeStore[displayedChapterId] = document.getElementById('code-editor').value;
  if (snap.theme === 'dark' || snap.theme === 'light') {
    document.documentElement.setAttribute('data-theme', snap.theme);
  }
  if (snap.solved && typeof snap.solved === 'object') {
    Object.keys(snap.solved).forEach(k => { solvedStore[k] = !!snap.solved[k]; });
  }
  if (snap.code && typeof snap.code === 'object') {
    CHAPTERS.forEach(({ id }) => { if (typeof snap.code[id] === 'string') userCodeStore[id] = snap.code[id]; });
  }
  const targetIdx = (Number.isInteger(snap.chapterIdx) && snap.chapterIdx >= 0 && snap.chapterIdx < CHAPTERS.length)
    ? snap.chapterIdx : currentChapterIdx;
  selectChapter(targetIdx, false);
  if (!silent) {
    setSyncStatus(`✅ 已成功恢复进度（已通关 ${Object.keys(solvedStore).filter(k => solvedStore[k]).length} 章，快照时间: ${snap.ts ? snap.ts.slice(0, 19).replace('T', ' ') : '刚刚'}）`);
  }
  return true;
}

function loadLocalState() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      applyStateSnapshot(JSON.parse(raw), true);
    }
    const gRaw = localStorage.getItem(GIST_CFG_KEY);
    if (gRaw) {
      const gCfg = JSON.parse(gRaw);
      if (gCfg.token) document.getElementById('gist-token-input').value = gCfg.token;
      if (gCfg.gistId) document.getElementById('gist-id-input').value = gCfg.gistId;
    }
  } catch (e) {}

  // 检查网址 Hash 是否带有跨设备同步码 #sync=...
  if (window.location.hash && window.location.hash.startsWith('#sync=')) {
    try {
      const b64 = decodeURIComponent(window.location.hash.slice(6));
      const jsonStr = decodeURIComponent(escape(atob(b64)));
      const snap = JSON.parse(jsonStr);
      applyStateSnapshot(snap, false);
      document.getElementById('sync-drawer').style.display = 'block';
      history.replaceState(null, '', window.location.pathname + window.location.search);
    } catch (e) {
      setSyncStatus('⚠️ URL 同步码解析失败', true);
    }
  }
}

function toggleTheme() {
  const cur = document.documentElement.getAttribute('data-theme') || 'dark';
  const next = cur === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  saveLocalState();
}

function toggleSyncDrawer() {
  const d = document.getElementById('sync-drawer');
  d.style.display = d.style.display === 'none' ? 'block' : 'none';
}

function buildSyncSnapshot(includeFullCode) {
  saveLocalState();
  const customCode = {};
  if (includeFullCode) {
    CHAPTERS.forEach(ch => {
      if (typeof userCodeStore[ch.id] === 'string' && userCodeStore[ch.id] !== ch.lab_code) {
        customCode[ch.id] = userCodeStore[ch.id];
      }
    });
  }
  return {
    v: 1,
    ts: new Date().toISOString(),
    chapterIdx: currentChapterIdx,
    theme: document.documentElement.getAttribute('data-theme') || 'dark',
    solved: solvedStore,
    code: customCode
  };
}

function copySyncUrl() {
  try {
    const snap = buildSyncSnapshot(true);
    const b64 = btoa(unescape(encodeURIComponent(JSON.stringify(snap))));
    const base = window.location.href.split('#')[0];
    const url = `${base}#sync=${encodeURIComponent(b64)}`;
    const input = document.getElementById('sync-url-preview');
    input.value = url;
    input.select();
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(url);
    } else {
      document.execCommand('copy');
    }
    setSyncStatus('🔗 跨设备同步链接已复制到剪贴板！发到手机/iPad 浏览器打开即可同步进度与代码。');
  } catch (e) {
    setSyncStatus('⚠️ 生成同步链接失败: ' + e.message, true);
  }
}

function importFromSyncCodePrompt() {
  const input = document.getElementById('sync-url-preview');
  const raw = (input.value && !input.value.startsWith('http://') && !input.value.startsWith('https://') && !input.value.startsWith('file://'))
    ? input.value.trim()
    : window.prompt('请粘贴跨设备同步链接或 #sync= 后的同步码：', '');
  if (!raw) return;
  try {
    const b64 = raw.includes('#sync=') ? decodeURIComponent(raw.split('#sync=')[1]) : decodeURIComponent(raw.trim());
    const snap = JSON.parse(decodeURIComponent(escape(atob(b64))));
    applyStateSnapshot(snap, false);
  } catch (e) {
    setSyncStatus('❌ 同步码格式无效，请确认复制完整。', true);
  }
}

function exportProgressJson() {
  const snap = buildSyncSnapshot(true);
  const blob = new Blob([JSON.stringify(snap, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `learn-minimind-progress-${new Date().toISOString().slice(0, 10)}.json`;
  a.click();
  setSyncStatus('📥 学习存档 .json 已导出下载！');
}

function importProgressJson(event) {
  const file = event.target.files && event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = e => {
    try {
      const snap = JSON.parse(e.target.result);
      applyStateSnapshot(snap, false);
    } catch (err) {
      setSyncStatus('❌ JSON 存档文件解析失败', true);
    }
  };
  reader.readAsText(file);
}

function saveGistConfig() {
  try {
    const token = document.getElementById('gist-token-input').value.trim();
    const gistId = document.getElementById('gist-id-input').value.trim();
    localStorage.setItem(GIST_CFG_KEY, JSON.stringify({ token, gistId }));
  } catch (e) {}
}

async function pushToGist() {
  saveGistConfig();
  const token = document.getElementById('gist-token-input').value.trim();
  let gistId = document.getElementById('gist-id-input').value.trim();
  if (!token) {
    setSyncStatus('⚠️ 请先填入带有 gist 权限的 GitHub Personal Access Token', true);
    return;
  }
  setSyncStatus('⏳ 正在上传学习进度与代码到 GitHub Gist...');
  const snap = buildSyncSnapshot(true);
  const body = {
    description: 'Learn-MiniMind Multi-Device Study Progress & Code Sync',
    public: false,
    files: {
      'learn-minimind-sync.json': { content: JSON.stringify(snap, null, 2) }
    }
  };
  try {
    const url = gistId ? `https://api.github.com/gists/${gistId}` : 'https://api.github.com/gists';
    const method = gistId ? 'PATCH' : 'POST';
    const res = await fetch(url, {
      method,
      headers: {
        'Accept': 'application/vnd.github+json',
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(body)
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    document.getElementById('gist-id-input').value = data.id;
    saveGistConfig();
    setSyncStatus(`☁️ 云端上传成功！Gist ID: ${data.id}（在其他设备填入此 Gist ID 点击「从云端拉取」即可同步）`);
  } catch (e) {
    setSyncStatus('❌ 上传 Gist 失败: ' + e.message, true);
  }
}

async function pullFromGist() {
  saveGistConfig();
  const token = document.getElementById('gist-token-input').value.trim();
  const gistId = document.getElementById('gist-id-input').value.trim();
  if (!gistId) {
    setSyncStatus('⚠️ 请填入要拉取的 Gist ID', true);
    return;
  }
  setSyncStatus('⏳ 正在从 GitHub Gist 拉取进度与代码...');
  try {
    const headers = { 'Accept': 'application/vnd.github+json' };
    if (token) headers['Authorization'] = `Bearer ${token}`;
    const res = await fetch(`https://api.github.com/gists/${gistId}`, { headers });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const fileObj = data.files && data.files['learn-minimind-sync.json'];
    if (!fileObj || !fileObj.content) throw new Error('Gist 中未找到 learn-minimind-sync.json');
    applyStateSnapshot(JSON.parse(fileObj.content), false);
  } catch (e) {
    setSyncStatus('❌ 从 Gist 拉取失败: ' + e.message, true);
  }
}

// 初始化
window.addEventListener('load', () => {
  initWorkspace();
  loadLocalState();
  if (!displayedChapterId) selectChapter(0);
  setTimeout(verifyAllChaptersLatex, 0);
  checkBackend();
});
</script>
</body>
</html>
"""
    out_path = ROOT / "index.html"
    html_doc = html_doc.replace("__KATEX_CSS__", katex_css).replace("__KATEX_JS__", katex_js).replace("__CHAPTERS_JSON__", chapters_json)
    for marker, filename in [("__WORKSPACE_CSS__", "workspace.css"), ("__WORKSPACE_JS__", "workspace.js")]:
        html_doc = html_doc.replace(marker, (ROOT / "web" / filename).read_text(encoding="utf-8"))
    out_path.write_text(html_doc, encoding="utf-8")
    print(f"已生成交互式网页课程与在线编程平台: {out_path} ({len(html_doc) / 1024:.1f} KB)")


if __name__ == "__main__":
    build_html()
