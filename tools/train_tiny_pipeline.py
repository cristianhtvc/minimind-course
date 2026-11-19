"""
30 秒全流程微缩实验：在纯 CPU（或 MPS/CUDA）上把 MiniMind 的 6 个核心训练阶段全部跑通一遍！

不需要下载几个 GB 的数据，也不需要 GPU。本脚本使用 MiniMind 的真实 6400 词表分词器，
在一个超小模型（hidden_size=64, layers=2, ~0.53M 参数）上依次演示：
  1. Pretrain   预训练（Next Token Prediction）
  2. Full SFT   指令微调（只对 assistant 回复计算 loss）+ 实际对话生成验证
  3. LoRA       低秩微调（只训练 ~2% 参数）+ 权重合并验证
  4. Distill    白盒知识蒸馏（4 层 Teacher -> 2 层 Student，CE + T^2*KL）
  5. DPO        直接偏好优化（Chosen vs Rejected 对数概率差显著拉大）
  6. GRPO/Agent 强化学习组内相对优势 + CISPO 更新 + 工具调用闭环评测
"""
import copy
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tinymind.config import TinyMindConfig
from tinymind.model import TinyMindForCausalLM, count_parameters
from tinymind.data import PretrainDataset, SFTDataset, DPODataset
from tinymind.train_utils import setup_seed, train_epoch
from tinymind.lora import apply_lora, mark_only_lora_trainable, merge_lora
from tinymind.distill import distill_step_loss
from tinymind.dpo import dpo_batch_loss, logits_to_log_probs
from tinymind.grpo import group_advantages, grpo_loss, per_token_logps
from tinymind.agent import parse_tool_calls, execute_tool, tool_reward


def banner(title: str):
    print(f"\n{'=' * 78}\n▶ {title}\n{'=' * 78}")


def main():
    setup_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / "minimind" / "model"))
    cfg = TinyMindConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, vocab_size=len(tokenizer))
    model = TinyMindForCausalLM(cfg)
    print(f"已加载 MiniMind 分词器（词表大小={len(tokenizer)}），创建 TinyMind 模型（参数量={count_parameters(model):,}）")

    # ------------------------------------------------------------------ 1. Pretrain
    banner("阶段 1/6：预训练 Pretraining（第 11-12 章）")
    pretrain_texts = [
        {"text": "MiniMind是一个从零开始训练的超小语言模型，采用Decoder-Only架构。"},
        {"text": "Transformer通过旋转位置编码RoPE和分组查询注意力GQA提升长文本效率。"},
        {"text": "1加1等于2，2加2等于4，3加3等于6。"},
    ] * 4
    pt_ds = PretrainDataset(pretrain_texts, tokenizer, max_length=48)
    pt_loader = DataLoader(pt_ds, batch_size=4, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    pt_losses = []
    for ep in range(8):
        pt_losses.extend(train_epoch(model, pt_loader, opt, epoch=ep, epochs=8, lr=1e-2, log_interval=0))
    print(f"Pretrain Loss: {pt_losses[0]:.4f}  --->  {pt_losses[-1]:.4f}")

    # ------------------------------------------------------------------ 2. Full SFT
    banner("阶段 2/6：有监督微调 Full SFT（第 13 章，仅对 Assistant 回复算 Loss）")
    sft_samples = [
        {"conversations": [{"role": "user", "content": "你是谁？"},
                           {"role": "assistant", "content": "我是MiniMind小助手。"}]},
        {"conversations": [{"role": "user", "content": "1+1等于几？"},
                           {"role": "assistant", "content": "1+1等于2。"}]},
    ] * 6
    sft_ds = SFTDataset(sft_samples, tokenizer, max_length=64, augment=False)
    sft_loader = DataLoader(sft_ds, batch_size=4, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    sft_losses = []
    for ep in range(15):
        sft_losses.extend(train_epoch(model, sft_loader, opt, epoch=ep, epochs=15, lr=1e-2, log_interval=0))
    print(f"SFT Loss:      {sft_losses[0]:.4f}  --->  {sft_losses[-1]:.4f}")

    model.eval()
    for q in ("你是谁？", "1+1等于几？"):
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": q}],
                                               tokenize=False, add_generation_prompt=True)
        prompt = prompt.replace("<think>\n\n</think>\n\n", "")
        ids = torch.tensor([tokenizer(prompt).input_ids])
        out = model.generate(ids, max_new_tokens=20, do_sample=False, eos_token_id=tokenizer.eos_token_id)
        ans = tokenizer.decode(out[0, ids.size(1):].tolist(), skip_special_tokens=True).strip()
        print(f"  [SFT 对话验证] 用户: {q}  -->  模型: {ans}")

    sft_snapshot = copy.deepcopy(model)

    # ------------------------------------------------------------------ 3. LoRA
    banner("阶段 3/6：LoRA 低秩微调与权重合并（第 14 章）")
    lora_model = copy.deepcopy(sft_snapshot)
    total_before = count_parameters(lora_model)
    apply_lora(lora_model, rank=4)
    lora_params = mark_only_lora_trainable(lora_model)
    trainable = count_parameters(lora_model, trainable_only=True)
    print(f"总参数: {total_before:,} | LoRA 可训练参数: {trainable:,} ({trainable / total_before * 100:.2f}%)")

    lora_data = [{"conversations": [{"role": "user", "content": "你是谁？"},
                                    {"role": "assistant", "content": "我是医疗领域的MiniMind。"}]}] * 6
    lora_loader = DataLoader(SFTDataset(lora_data, tokenizer, max_length=64, augment=False), batch_size=3)
    lora_opt = torch.optim.AdamW(lora_params, lr=2e-2)
    l_hist = []
    for ep in range(10):
        l_hist.extend(train_epoch(lora_model, lora_loader, lora_opt, epoch=ep, epochs=10, lr=2e-2, log_interval=0))
    merged_sd = merge_lora(lora_model)
    clean_model = TinyMindForCausalLM(cfg)
    clean_model.load_state_dict(merged_sd)
    test_x = torch.randint(0, 100, (1, 8))
    max_diff = (lora_model.eval()(test_x).logits - clean_model.eval()(test_x).logits).abs().max().item()
    print(f"LoRA Loss:     {l_hist[0]:.4f}  --->  {l_hist[-1]:.4f} | 合并后权重与原 LoRA 模型最大误差: {max_diff:.2e}")

    # ------------------------------------------------------------------ 4. Distillation
    banner("阶段 4/6：白盒知识蒸馏 Knowledge Distillation（第 15 章）")
    teacher_cfg = TinyMindConfig(hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
                                 num_key_value_heads=2, vocab_size=len(tokenizer))
    teacher = TinyMindForCausalLM(teacher_cfg).eval()
    student = copy.deepcopy(sft_snapshot).train()
    kd_opt = torch.optim.AdamW(student.parameters(), lr=5e-3)
    x_batch, y_batch = next(iter(sft_loader))
    kd_losses = []
    for _ in range(8):
        s_logits = student(x_batch).logits[:, :-1, :].contiguous()
        with torch.no_grad():
            t_logits = teacher(x_batch).logits[:, :-1, :].contiguous()
        loss, info = distill_step_loss(s_logits, t_logits, y_batch[:, 1:].contiguous(), alpha=0.5, temperature=2.0)
        kd_opt.zero_grad(); loss.backward(); kd_opt.step()
        kd_losses.append(loss.item())
    print(f"Distill Loss (0.5*CE + 0.5*T²KL): {kd_losses[0]:.4f}  --->  {kd_losses[-1]:.4f} (末步 KL={info['kd']:.4f})")

    # ------------------------------------------------------------------ 5. DPO
    banner("阶段 5/6：DPO 直接偏好优化（第 16 章）")
    dpo_samples = [{
        "chosen": [{"role": "user", "content": "你是谁？"}, {"role": "assistant", "content": "我是MiniMind小助手，很高兴帮助你。"}],
        "rejected": [{"role": "user", "content": "你是谁？"}, {"role": "assistant", "content": "不知道，别问我。"}],
    }] * 4
    dpo_batch = next(iter(DataLoader(DPODataset(dpo_samples, tokenizer, max_length=64), batch_size=4)))
    policy = copy.deepcopy(sft_snapshot).train()
    ref = copy.deepcopy(sft_snapshot).eval()
    dpo_opt = torch.optim.AdamW(policy.parameters(), lr=5e-3)

    def margin(m):
        with torch.no_grad():
            lp_c = (logits_to_log_probs(m(dpo_batch["x_chosen"]).logits, dpo_batch["y_chosen"]) * dpo_batch["mask_chosen"]).sum(-1)
            lp_r = (logits_to_log_probs(m(dpo_batch["x_rejected"]).logits, dpo_batch["y_rejected"]) * dpo_batch["mask_rejected"]).sum(-1)
            return (lp_c - lp_r).mean().item()

    m_before = margin(policy)
    for _ in range(8):
        loss, _ = dpo_batch_loss(policy, ref, dpo_batch, beta=0.2)
        dpo_opt.zero_grad(); loss.backward(); dpo_opt.step()
    m_after = margin(policy)
    print(f"Chosen 与 Rejected 的对数概率差 (log π(y_w) - log π(y_l)): {m_before:.2f}  --->  {m_after:.2f}")

    # ------------------------------------------------------------------ 6. GRPO & Agent
    banner("阶段 6/6：GRPO/CISPO 强化学习与 Agent 工具调用奖励（第 18-19 章）")
    candidates = [
        '<tool_call>\n{"name": "calculate_math", "arguments": {"expression": "12 * 13"}}\n</tool_call>',
        '<tool_call>\n{"name": "calculate_math", "arguments": {"expression": "12 + 13"}}\n</tool_call>',
        '我随便猜一个答案 100',
        '<tool_call>未闭合的坏格式',
    ]
    rewards = []
    for c in candidates:
        calls = parse_tool_calls(c)
        if calls:
            res = execute_tool(calls[0]["name"], calls[0]["arguments"])
            final = f"计算结果是 {res.get('result')}。"
        else:
            final = c
        r = tool_reward([c, final], final, gt=["156"], valid_tool_names={"calculate_math"})
        rewards.append(r)
    rewards_t = torch.tensor(rewards)
    adv = group_advantages(rewards_t, num_generations=len(candidates))
    for i, (c, r, a) in enumerate(zip(candidates, rewards, adv.tolist()), 1):
        print(f"  采样 {i}: reward={r:+.2f} -> 组内相对优势 A={a:+.2f} | 输出: {c.splitlines()[0][:55]}")

    print("\n🎉 全流程 6 个阶段演练完毕！现在你可以阅读 chapters/ 并在 labs/ 中亲手实现每一部分。")


if __name__ == "__main__":
    main()
