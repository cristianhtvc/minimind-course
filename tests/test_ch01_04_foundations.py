"""第 1-4 章测试：PyTorch/LM 基础、Byte-Level BPE 分词器、RMSNorm"""
import math
import torch
import torch.nn.functional as F

from conftest import load_module
from model.model_minimind import RMSNorm as MiniMindRMSNorm


def test_ch01_02_basics():
    basics = load_module("basics")
    torch.manual_seed(0)
    logits = torch.randn(8, 20) * 50  # 故意放大以测试数值稳定性
    probs = basics.stable_softmax(logits, dim=-1)
    assert torch.allclose(probs, F.softmax(logits, dim=-1), atol=1e-6)
    assert not torch.isnan(basics.stable_softmax(torch.tensor([[1000.0, 1001.0]]))).any()

    targets = torch.tensor([1, 3, -100, 0, 5, -100, 2, 4])
    ce = basics.cross_entropy(logits, targets, ignore_index=-100)
    ref = F.cross_entropy(logits, targets, ignore_index=-100)
    assert torch.allclose(ce, ref, atol=1e-6)
    assert abs(basics.perplexity(float(ce)) - math.exp(float(ref))) < 1e-4


def test_ch03_bpe_roundtrip_and_compression():
    bpe_mod = load_module("tokenizer_bpe")
    assert bpe_mod.get_stats([1, 2, 3, 1, 2]) == {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    assert bpe_mod.merge([1, 2, 3, 1, 2], (1, 2), 99) == [99, 3, 99]

    corpus = ("人工智能与大语言模型正在改变世界。Large language models learn next token prediction. " * 20)
    tok = bpe_mod.ByteLevelBPE(special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"])
    tok.train(corpus, vocab_size=320)
    assert len(tok) == 320

    sample = "<|im_start|>user\n大语言模型是什么？Large language models!<|im_end|>"
    ids = tok.encode(sample)
    # 合并后 token 数必须显著少于原始 UTF-8 字节数
    assert len(ids) < len(sample.encode("utf-8")) * 0.65
    # 特殊 token 应该被整体编码为单个 id
    assert tok.special_tokens["<|im_start|>"] in ids
    assert tok.special_tokens["<|im_end|>"] in ids
    # 编解码完全可逆
    assert tok.decode(ids) == sample


def test_ch04_rmsnorm_matches_minimind():
    norm_mod = load_module("norm")
    torch.manual_seed(42)
    x = torch.randn(2, 16, 64, dtype=torch.bfloat16, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)

    ours = norm_mod.RMSNorm(64, eps=1e-6)
    ref = MiniMindRMSNorm(64, eps=1e-6)
    with torch.no_grad():
        ours.weight.copy_(torch.randn(64))
        ref.weight.copy_(ours.weight)

    y = ours(x)
    y_ref = ref(x_ref)
    assert y.dtype == torch.bfloat16
    assert torch.allclose(y, y_ref, atol=1e-5)
    y.sum().backward()
    y_ref.sum().backward()
    assert torch.allclose(x.grad, x_ref.grad, atol=1e-5)
