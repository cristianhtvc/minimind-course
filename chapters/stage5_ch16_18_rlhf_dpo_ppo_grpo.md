# 第五阶段（第 16–18 章）：强化学习后训练全解析 —— DPO、PPO 与 GRPO / CISPO

> **本阶段目标**：
> SFT 只教会了模型「模仿正例的下一个词」，却没有告诉模型「什么回答比什么回答更好、什么格式是错的、哪道数学题算对了」。本阶段将带你站在 MiniMind 作者提出的 **策略优化（Policy Optimization, PO）统一视角** 上，一口气打通当代大模型后训练（Post-Training）的三大主流强化学习算法：
> 1. **离线偏好优化 DPO**（第 16 章，`train_dpo.py`）
> 2. **经典在线 Actor-Critic 算法 PPO + GAE**（第 17 章，`train_ppo.py`）
> 3. **前沿无 Critic 在线算法 GRPO（DeepSeek）与 CISPO（MiniMax）+ SGLang Rollout 引擎**（第 18 章，`train_grpo.py` + `rollout_engine.py`）
>
> **配套代码与练习**：
> - 参考实现：[tinymind/dpo.py](../tinymind/dpo.py)、[tinymind/ppo.py](../tinymind/ppo.py)、[tinymind/grpo.py](../tinymind/grpo.py)
> - 动手练习：[labs/dpo.py](../labs/dpo.py)（`TODO(ch16-1)` ~ `TODO(ch16-2)`）、[labs/ppo.py](../labs/ppo.py)（`TODO(ch17-1)` ~ `TODO(ch17-3)`）、[labs/grpo.py](../labs/grpo.py)（`TODO(ch18-1)` ~ `TODO(ch18-2)`）
> - 对照 MiniMind 源码：[minimind/trainer/train_dpo.py](../minimind/trainer/train_dpo.py)、[minimind/trainer/train_ppo.py](../minimind/trainer/train_ppo.py)、[minimind/trainer/train_grpo.py](../minimind/trainer/train_grpo.py)、[minimind/trainer/rollout_engine.py](../minimind/trainer/rollout_engine.py)
> - 闯关评测：`python tools/grade.py ch16`、`python tools/grade.py ch17`

---

## 16.0 所有策略优化（xxPO）算法的统一视角

在阅读三个 RL 脚本前，先记住 MiniMind README 中总结的统一公式——所有 Policy Optimization 算法都只是在最大化同一个目标函数（代码中最小化其相反数 $\mathcal{L}_{\text{PO}} = -\mathcal{J}_{\text{PO}}$）：

$$\mathcal{J}_{\text{PO}}(\theta) = \mathbb{E}_{q \sim P(Q),\, o \sim \pi(\cdot \mid q)} \left[ \underbrace{\Phi(r_t(\theta),\, A_t)}_{\text{① 策略更新项}} - \underbrace{h\big(\mathrm{KL}(\pi_\theta \parallel \pi_{\text{ref}})\big)}_{\text{③ KL 正则项}} \right]$$

其中 $r_t(\theta) = \frac{\pi_\theta(o_t \mid q, o_{<t})}{\pi_{\text{old}}(o_t \mid q, o_{<t})}$ 是新旧策略在 token $t$ 上的**重要性采样概率比**，$A_t$ 是**优势函数（Advantage）**。
**DPO、PPO、GRPO、CISPO 的全部区别，仅仅是填表方式不同**：

| 算法 | 在线/离线 | 需要几个模型驻留显存？ | ① 策略项 $\Phi(r_t, A_t)$ | ② 优势项 $A_t$ 怎么来？ | ③ KL 正则项 $h(\mathrm{KL})$ |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **DPO** | Off-policy（静态偏好对） | **2 个**：Actor $\pi_\theta$ + Ref $\pi_{\text{ref}}$ | $\log\sigma\big(\beta(\log\frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - \log\frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)})\big)$ | 隐含在 $y_w$（好）与 $y_l$（差）的对比中 | 隐含在参考模型比值与 $\beta$ 中 |
| **PPO** | On-policy（实时采样） | **4 个**：Actor + **Critic $V_\phi$** + Ref + Reward | $\min\big(r_t A_t,\, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t\big)$ | Critic 预测价值 $V_t$，用 **GAE** 逐 token 倒推 $A_t$ | $\beta \cdot (e^{\Delta} - \Delta - 1)$ |
| **GRPO** | On-policy（每题采 $G$ 条） | **3 个**：Actor + Ref + Reward（**砍掉 Critic！**） | $\min\big(r_t A_i,\, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_i\big)$ | 同一问题 $G$ 个回答的**组内标准化奖励** $A_i = \frac{R_i - \mu_G}{\sigma_G}$ | $\beta \cdot (e^{\Delta} - \Delta - 1)$ |
| **CISPO** | On-policy（每题采 $G$ 条） | **3 个**：Actor + Ref + Reward | $\text{sg}\big(\min(r_t, \epsilon_{\text{high}})\big) \cdot A_i \cdot \log\pi_\theta(o_t)$ | 与 GRPO 相同（组内相对奖励） | $\beta \cdot (e^{\Delta} - \Delta - 1)$ |

---

## 第 16 章：DPO 直接偏好优化（`train_dpo.py`）

### 16.1 DPO 的数学推导：为什么可以不要 Reward Model 和 Critic？

传统 RLHF 需要两步：① 先用人类偏好对 $(x, y_w \succ y_l)$ 按 Bradley-Terry 模型 $P(y_w \succ y_l \mid x) = \sigma\big(r(x, y_w) - r(x, y_l)\big)$ 训练一个奖励模型 $r(x, y)$；② 再用 PPO 最大化 $\mathbb{E}[r(x, y)] - \beta \mathrm{KL}(\pi_\theta \parallel \pi_{\text{ref}})$。
Rafailov et al. (2023, *DPO*) 证明：带 KL 约束的强化学习目标存在**解析最优解**，反解出奖励函数 $r(x, y)$ 与最优策略 $\pi^*(y \mid x)$ 的关系为：

$$r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$

把上式直接代回 Bradley-Terry 模型，归一化常数 $\log Z(x)$ 在 $r(x, y_w) - r(x, y_l)$ 中**完美抵消**！于是得到直接优化策略 $\pi_\theta$ 的 DPO 损失：

$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \left[ \left(\log \pi_\theta(y_w \mid x) - \log \pi_{\text{ref}}(y_w \mid x)\right) - \left(\log \pi_\theta(y_l \mid x) - \log \pi_{\text{ref}}(y_l \mid x)\right) \right] \right) \right]$$

其中序列的对数概率等于**仅在 Assistant 回复位置（`mask == 1`）上**所有 token 对数概率之和：
$$\log \pi(y \mid x) = \sum_{t=1}^{|y|} \text{mask}_t \cdot \log \pi(y_t \mid x, y_{<t})$$

### 16.2 对照 MiniMind 源码：`train_dpo.py` 逐行精读

打开 [minimind/trainer/train_dpo.py#L25-L84](../minimind/trainer/train_dpo.py#L25-L84)：
1. **`logits_to_log_probs(logits, labels)`（第 25–31 行）**：
   ```python
   log_probs = F.log_softmax(logits, dim=2)
   log_probs_per_token = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
   ```
   `logits` 形状是 `[B, L, V]`，`labels` 形状是 `[B, L]`（在 `DPODataset` 中已经预先做过了 `x = ids[:-1], y = ids[1:]` 错位！）。`torch.gather` 在词表维 `dim=2` 上精准取出真实目标 token 对应的 log 概率。
2. **Chosen 与 Rejected 拼接成单次 `2B` 前向传播（第 63–65, 39–44 行）**：
   为了避免对 `chosen` 和 `rejected` 分两次调用 `model(x)`（两次 Kernel 启动慢，且在 DDP 下两次 forward 会触发额外的桶同步问题），MiniMind 在 `dim=0` 上把它们拼成一个 `[2B, L]` 的大张量：
   ```python
   x = torch.cat([x_chosen, x_rejected], dim=0)
   ```
   算完序列级 `ref_log_probs` 和 `policy_log_probs` 后，在 `dpo_loss` 里直接按前半 `[:batch_size // 2]`（Chosen）和后半 `[batch_size // 2:]`（Rejected）切开即可！

---

## 第 17 章：RLAIF 与 PPO 近端策略优化（`train_ppo.py`）

DPO 简单稳定，但它是 **Off-Policy** 的：模型只在静态数据集的 `chosen/rejected` 上打转，没有自己去生成和试错。要让模型在推理、自我纠错上产生质变，需要 **On-Policy 在线强化学习**。

### 17.1 PPO 的四模型协同与 `CriticModel`

在 [minimind/trainer/train_ppo.py](../minimind/trainer/train_ppo.py) 中，同时驻留了 4 个角色：
1. **`actor_model` ($\pi_\theta$)**：正在被训练的对话模型（初始化自 `full_sft`）。
2. **`ref_model` ($\pi_{\text{ref}}$)**：冻结的 `full_sft` 拷贝（`eval()` + `requires_grad_(False)`），用来算 KL 散度防止 Actor 训崩。
3. **`reward_model` ($R$)**：裁判（MiniMind 支持接入外部 `internlm2-1_8b-reward` + 长度/思考标签/重复惩罚规则分，见第 51–68 行）。
4. **`critic_model` ($V_\phi$，第 36–48 行)**：价值网络，继承自 `MiniMindForCausalLM`，把输出层换成 `self.value_head = nn.Linear(hidden_size, 1)`，对序列的每一个位置 $t$ 预测一个标量价值 $V_t$（代表「从当前 token 往后写完，预计最终能拿多少分」）。

### 17.2 稀疏奖励放置与 GAE（广义优势估计）倒序递推

整条回答生成完后，Reward Model 只给**整条回答打一个总分 $R$**。但在生成过程中，模型走了 $T$ 步（吐了 $T$ 个 token），怎么把这 1 个总分分配给每一个 token？
查看 [train_ppo.py#L140-L152](../minimind/trainer/train_ppo.py#L140-L152)：
1. **末尾放置稀疏奖励**：
   构造全零张量 `token_rewards`（形状 `[B, T]`），只把外部总奖励 $R$ 加在**每条回答的最后一个有效 token `last_idx = resp_lengths - 1`** 上！即 $r_t = 0\;(t < T-1),\; r_{T-1} = R$。
2. **GAE（Generalized Advantage Estimation，Schulman et al. 2016）从后往前递推**：
   定义单步 TD 误差（Temporal-Difference Error）：
   $$\delta_t = r_t + \gamma V_{t+1} - V_t \qquad (\text{其中末尾 } V_T = 0)$$
   然后从最后一个 token $t = T-1$ 往回递推到 $t = 0$：
   $$A_t = \delta_t + \gamma \lambda A_{t+1}$$
   ```python
   gen_len = old_resp_values.size(1); lastgaelam = torch.zeros(B, device=args.device); advs_rev = []
   for t in reversed(range(gen_len)):
       nv = old_resp_values[:, t + 1] if t < gen_len - 1 else 0.0
       delta = token_rewards[:, t] + args.gamma * nv - old_resp_values[:, t]
       lastgaelam = delta + args.gamma * args.lam * lastgaelam
       advs_rev.append(lastgaelam)
   advantages = torch.stack(advs_rev[::-1], dim=1)
   returns = advantages + old_resp_values
   ```
   - **直觉理解**：如果到第 $t$ 个 token 时 Critic 预测还能拿 $V_t = 1.0$ 分，结果模型吐出第 $t$ 个词后，下一步 Critic 发现局势大好，预测变成了 $V_{t+1} = 2.5$ 分，那么 $\delta_t = 2.5 - 1.0 = +1.5 > 0$，说明**第 $t$ 个 token 是一个超预期的神来之笔**，应该给正优势鼓励！

### 17.3 PPO-Clip 双重裁剪、k3 KL 估计器与 DDP 早停防死锁

查看 [train_ppo.py#L194-L232](../minimind/trainer/train_ppo.py#L194-L232)：
1. **策略裁剪（Policy Clip）**：
   在同一个 Rollout 批次上做多轮小批量更新（`ppo_update_iters`）时，新策略与采样时的旧策略概率比 $r_t(\theta) = \exp(\log\pi_\theta - \log\pi_{\text{old}})$ 会逐渐偏离 $1.0$。当 $A_t > 0$ 时如果 $r_t > 1 + \epsilon$，或者当 $A_t < 0$ 时如果 $r_t < 1 - \epsilon$，说明这一步迈得太大了，直接用 `torch.clamp(ratio, 1 - eps, 1 + eps)` 截断梯度！
2. **价值裁剪（Value Clip，第 212–215 行）**：同样限制新价值预测 `mb_resp_values` 偏离采样时的 `old_resp_values` 不超过 `cliprange_value`。
3. **Schulman k3 无偏正定 KL 估计器（第 206 行）**：
   如果直接用 $\log\pi_\theta - \log\pi_{\text{ref}}$ 近似 KL 散度，单个采样点上可能出现负数，方差很大。MiniMind 使用了 John Schulman 推荐的 $k_3$ 估计器（令 $\Delta = \log\pi_{\text{ref}} - \log\pi_\theta$）：
   $$\widehat{\mathrm{KL}}_{k_3} = e^{\Delta} - \Delta - 1 \ge 0$$
   由于对任意实数 $x$ 都有 $e^x \ge 1 + x$，该估计量**在每一个 token 上都严格非负**，且期望严格等于真实的 $\mathrm{KL}(\pi_\theta \parallel \pi_{\text{ref}})$！
4. **DDP 早停防死锁（第 196–226 行）**：
   当 `approx_kl > early_stop_kl` 时需要早停。在多卡 DDP 下，**绝不能某张卡自己 `break` 跳出循环**（否则其他还在跑 `backward()` 的卡会在等待 All-Reduce 时永久死锁）。MiniMind 做了两手保护：① 先用 `dist.all_reduce(approx_kl_val, op=AVG)` 让所有卡看到完全相同的 `approx_kl_val`；② 触发 `stop_ppo` 时该轮仍然把 `loss * 0.0` 走完 `backward()` 闭环，下一轮大家一起 `break`！

---

## 第 18 章：GRPO 与 CISPO + Rollout 采样引擎（`train_grpo.py`）

### 18.1 GRPO：为什么能直接砍掉 Critic 模型？

在 PPO 中，`critic_model` 的参数量和 `actor_model` 一样大（还要加上它自己的 AdamW 优化器状态），几乎占了一半显存，而且在小步数训练中 Critic 很难训准。
DeepSeekMath / DeepSeek-R1 提出的 **GRPO（Group Relative Policy Optimization，Shao et al. 2024）** 给出了一个极简方案：
- 对每一个问题 $q$，让策略模型连续采样 **$G$ 个不同的回答** $\{o_1, o_2, \dots, o_G\}$（MiniMind 默认 `num_generations = 6`），分别拿到奖励 $\{R_1, R_2, \dots, R_G\}$。
- 直接用**这 $G$ 个回答的组内平均分 $\mu_G$ 作为基线（Baseline），组内标准差 $\sigma_G$ 做归一化**，算出每条回答的优势 $A_i$：

$$A_i = \frac{R_i - \mathrm{mean}(R_1, \dots, R_G)}{\mathrm{std}(R_1, \dots, R_G) + 10^{-4}}$$

对应 [train_grpo.py#L121-L124](../minimind/trainer/train_grpo.py#L121-L124) 仅仅 4 行代码：
```python
grouped_rewards = rewards.view(-1, args.num_generations)  # [B, num_gen]
mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)  # [B*num_gen]
std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)  # [B*num_gen]
advantages = (rewards - mean_r) / (std_r + 1e-4)  # [B*num_gen]
```
**直觉为什么这么有效？**
如 MiniMind README 所说：「大模型算对定积分也不足为奇，小模型算对加减法优势就应该是正的。」组内相对奖励自动适应了每一道题对当前模型而言的**相对难度**——在同一道题的 6 次尝试里，比平均水平好的那几次尝试拿正优势（增加概率），比平均水平差的那几次拿负优势（压低概率），完全不需要 Critic！

### 18.2 从 GRPO 到 CISPO（MiniMax-M1, 2025）

查看 [train_grpo.py#L134-L143](../minimind/trainer/train_grpo.py#L134-L143)：
```python
ratio = torch.exp(per_token_logps - old_per_token_logps)  # [B*num_gen, R]
if args.loss_type == "cispo":
    clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()
    per_token_loss = -(clamped_ratio * advantages.unsqueeze(1) * per_token_logps - args.beta * per_token_kl)
else:
    clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
    per_token_loss1 = ratio * advantages.unsqueeze(1)
    per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
    per_token_loss = -(torch.min(per_token_loss1, per_token_loss2) - args.beta * per_token_kl)
```
**为什么 MiniMind 默认使用 `loss_type = "cispo"`？**
在长思维链（`<think>...</think>`）强化学习中，那些真正引起反思转折的关键 token（比如 `"不过"`、`"等等"`、`"重新检查一下"`）原本在模型里的生成概率往往很低。一旦被采样出来并带来了高奖励，下一步它的概率比 $r_t = \pi_\theta / \pi_{\text{old}}$ 很容易瞬间超过 $1 + \epsilon = 1.2$。
- 在传统 PPO/GRPO 的 `min(r*A, clip(r, 1-ε, 1+ε)*A)` 中，一旦 $r_t > 1.2$，该 token 就会落入常数裁剪区，**梯度直接变成 0（被彻底丢弃）**！
- **CISPO（Clipped Importance Sampling Policy Optimization）** 改为对重要性采样权重做带上限的 stop-gradient 裁剪：
  $$\mathcal{L}_t^{\text{CISPO}} = -\mathrm{sg}\big(\min(r_t,\, \epsilon_{\text{high}})\big) \cdot A_t \cdot \log \pi_\theta(o_t \mid q, o_{<t}) + \beta \cdot \mathrm{KL}_t$$
  这样即使某个稀缺反思 token 的 $r_t$ 很大，它的权重也只是被限制在 $\epsilon_{\text{high}} = 5.0$，**梯度 $\nabla_\theta \log \pi_\theta$ 依然完整保留**！

### 18.3 Rollout 引擎：为什么 RL 训练需要 SGLang？（`rollout_engine.py`）

在 On-Policy RL（PPO / GRPO / Agent RL）中，**80% 以上的时间都花在自回归采样（Rollout）上**。
MiniMind 在 [trainer/rollout_engine.py](../minimind/trainer/rollout_engine.py) 中设计了统一的 `RolloutEngine` 抽象接口：
- **`TorchRolloutEngine`**：直接调用 `policy_model.generate()`，零外部依赖，开箱即用。
- **`SGLangRolloutEngine`**：把采样请求通过 HTTP 发给本地启动的 **SGLang** 推理服务器（自带 RadixAttention 前缀缓存与连续批处理 Continuous Batching，同一个 Prompt 采样 $G=6$ 条回答时 Prompt 的 KV Cache 100% 共享！）；每隔 `save_interval` 步，训练进程调用 `rollout_engine.update_policy(model)` 将最新参数保存为 HuggingFace 格式，并调 SGLang 的 `/update_weights_from_disk` + `/flush_cache` 接口**在线热更新推理引擎权重**！

---

## 🛠️ 本阶段动手实验（Labs）

```bash
# 1. 完成 labs/dpo.py 中的 TODO(ch16-1) ~ TODO(ch16-2)
python tools/grade.py ch16 -v

# 2. 完成 labs/ppo.py 中的 TODO(ch17-1) ~ TODO(ch17-3)
#    以及 labs/grpo.py 中的 TODO(ch18-1) ~ TODO(ch18-2)
python tools/grade.py ch17 -v
```
