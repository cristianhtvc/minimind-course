"""第 11-19 章测试：数据集、训练循环、LoRA、知识蒸馏、DPO、PPO、GRPO/CISPO、Agent 工具调用"""
import copy
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from conftest import load_module, ROOT
from tinymind.config import TinyMindConfig
from tinymind.model import TinyMindForCausalLM

TOKENIZER_DIR = str(ROOT / "minimind" / "model")


def test_ch11_13_datasets_match_minimind():
    data_mod = load_module("data")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

    pt = data_mod.PretrainDataset([{"text": "清晨的阳光透过窗帘洒进房间。"}], tokenizer, max_length=32)
    x, y = pt[0]
    assert x.shape == (32,) and y.shape == (32,)
    assert x[0].item() == tokenizer.bos_token_id
    assert (y[x == tokenizer.pad_token_id] == -100).all()
    assert (y[x != tokenizer.pad_token_id] == x[x != tokenizer.pad_token_id]).all()

    convs = [{"conversations": [
        {"role": "user", "content": "1+1等于几？"},
        {"role": "assistant", "content": "等于2。"},
        {"role": "user", "content": "再加1呢？"},
        {"role": "assistant", "content": "等于3。"},
    ]}]
    sft = data_mod.SFTDataset(convs, tokenizer, max_length=128, augment=False)
    sx, sy = sft[0]
    # 只有 assistant 回复与 <|im_end|>\n 位置有 label，user 问题部分必须全是 -100
    supervised_text = tokenizer.decode([t for t in sy.tolist() if t != -100])
    assert "等于2。" in supervised_text and "等于3。" in supervised_text
    assert "1+1等于几" not in supervised_text and "再加1呢" not in supervised_text


def test_ch12_train_loop_reduces_loss():
    train_mod = load_module("train_utils")
    assert abs(train_mod.get_lr(0, 100, 1e-3) - 1e-3) < 1e-9
    assert abs(train_mod.get_lr(100, 100, 1e-3) - 1e-4) < 1e-9

    torch.manual_seed(0)
    cfg = TinyMindConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, vocab_size=64)
    model = TinyMindForCausalLM(cfg)
    seq = torch.randint(1, 60, (4, 16))
    loader = DataLoader(list(zip(seq, seq.clone())), batch_size=2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    hist = []
    for ep in range(25):
        hist.extend(train_mod.train_epoch(model, loader, opt, epoch=ep, epochs=25, lr=1e-2,
                                          accumulation_steps=2, log_interval=0))
    assert hist[-1] < hist[0] * 0.5


def test_ch14_lora_zero_init_and_merge():
    lora_mod = load_module("lora")
    torch.manual_seed(0)
    cfg = TinyMindConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=4,
                         num_key_value_heads=2, vocab_size=64)
    model = TinyMindForCausalLM(cfg).eval()
    x = torch.randint(0, 64, (2, 8))
    y_before = model(x).logits.detach()

    lora_mod.apply_lora(model, rank=4)
    params = lora_mod.mark_only_lora_trainable(model)
    assert len(params) > 0
    assert all("lora" in n for n, p in model.named_parameters() if p.requires_grad)
    # B 零初始化 -> 刚挂上 LoRA 时输出与原模型完全一致
    assert torch.allclose(model(x).logits, y_before, atol=1e-6)

    # 给 B 随机赋值（模拟训练后），验证 merge_lora 后的独立模型输出与挂着 LoRA 的模型完全一致
    for p in params:
        p.data.normal_(0, 0.1)
    y_lora = model(x).logits.detach()
    merged_sd = lora_mod.merge_lora(model)
    clean_model = TinyMindForCausalLM(cfg).eval()
    clean_model.load_state_dict(merged_sd)
    assert torch.allclose(clean_model(x).logits, y_lora, atol=1e-5)


def test_ch15_distillation_loss():
    distill_mod = load_module("distill")
    logits = torch.randn(10, 32)
    # 学生 == 老师 时 KL 为 0
    assert distill_mod.distillation_loss(logits, logits, temperature=2.0).item() < 1e-6
    s = torch.randn(2, 6, 32, requires_grad=True)
    t = torch.randn(2, 6, 32)
    labels = torch.randint(0, 32, (2, 6))
    labels[:, 0] = -100
    loss, info = distill_mod.distill_step_loss(s, t, labels, alpha=0.3, temperature=2.0)
    loss.backward()
    assert s.grad is not None and info["kd"] > 0


def test_ch16_dpo_increases_chosen_margin():
    dpo_mod = load_module("dpo")
    torch.manual_seed(0)
    cfg = TinyMindConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=4,
                         num_key_value_heads=2, vocab_size=64)
    policy = TinyMindForCausalLM(cfg)
    ref = copy.deepcopy(policy).eval()
    batch = {
        "x_chosen": torch.tensor([[1, 2, 3, 4]]), "y_chosen": torch.tensor([[2, 3, 4, 5]]),
        "mask_chosen": torch.tensor([[0, 0, 1, 1]]),
        "x_rejected": torch.tensor([[1, 2, 8, 9]]), "y_rejected": torch.tensor([[2, 8, 9, 5]]),
        "mask_rejected": torch.tensor([[0, 0, 1, 1]]),
    }
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-2)
    losses = []
    for _ in range(5):
        loss, _ = dpo_mod.dpo_batch_loss(policy, ref, batch, beta=0.5)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


def test_ch17_18_ppo_and_grpo():
    ppo_mod = load_module("ppo")
    grpo_mod = load_module("grpo")

    # GAE: gamma=1, lam=1 且 V=0 时，每个位置的 advantage 就是从该位置到末尾的奖励和
    r = torch.tensor([[0.0, 0.0, 2.0]])
    v = torch.zeros_like(r)
    adv, ret = ppo_mod.compute_gae(r, v, gamma=1.0, lam=1.0)
    assert torch.allclose(adv, torch.tensor([[2.0, 2.0, 2.0]]))
    assert torch.allclose(ret, adv)

    logp = torch.randn(4, 6, requires_grad=True)
    old_logp = logp.detach() - 0.05
    ref_logp = logp.detach() - 0.1
    mask = torch.ones(4, 6)
    ploss, clipfrac = ppo_mod.ppo_policy_loss(logp, old_logp, torch.ones(4, 6), mask)
    vloss = ppo_mod.ppo_value_loss(torch.ones(4, 6), torch.zeros(4, 6), torch.ones(4, 6), mask)
    assert ploss.requires_grad and vloss.item() >= 0 and 0.0 <= clipfrac.item() <= 1.0

    # GRPO 组内优势：每组的均值必须为 0
    rewards = torch.tensor([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
    g_adv = grpo_mod.group_advantages(rewards, num_generations=3)
    assert torch.allclose(g_adv[:3].mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(g_adv[3:].mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(g_adv[:3], g_adv[3:], atol=1e-3)  # 组内相对排名相同 -> 优势相同

    for lt in ("grpo", "cispo"):
        gl = grpo_mod.grpo_loss(logp, old_logp, ref_logp, torch.tensor([1.0, -1.0, 0.5, -0.5]), mask, loss_type=lt)
        assert gl.requires_grad


def test_ch19_agent_tool_call():
    agent_mod = load_module("agent")
    text = '<think>\n算一下\n</think>\n<tool_call>\n{"name": "calculate_math", "arguments": {"expression": "256 * 37"}}\n</tool_call>'
    calls = agent_mod.parse_tool_calls(text)
    assert calls == [{"name": "calculate_math", "arguments": {"expression": "256 * 37"}}]
    res = agent_mod.execute_tool(calls[0]["name"], calls[0]["arguments"])
    assert res == {"result": "9472"}

    r_good = agent_mod.tool_reward([text, "答案是 9,472。"], "答案是 9,472。", ["9472"], {"calculate_math"})
    r_bad = agent_mod.tool_reward(["<tool_call>broken"], "不知道", ["9472"], {"calculate_math"})
    assert r_good == 3.0 and r_bad < 0
