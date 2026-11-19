"""
交互式闯关评测器：按章节检查你在 labs/ 中的实现进度（单次启动 pytest，约 3 秒完成全部 15 关检测）。

用法：
    python tools/grade.py              # 评测你在 labs/ 中的当前进度
    python tools/grade.py --ref        # 评测 tinymind/ 参考实现（应全部通过）
    python tools/grade.py ch05         # 只评测指定章节（支持 ch01..ch19）
    python tools/grade.py ch05 -v      # 查看该章节失败测试的详细堆栈
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CHAPTERS = [
    ("ch01-02", "PyTorch 与语言模型基础",   "basics.py",        "minimind/model/model_minimind.py",
     "tests/test_ch01_04_foundations.py::test_ch01_02_basics"),
    ("ch03",    "Byte-Level BPE 分词器",    "tokenizer_bpe.py", "minimind/trainer/train_tokenizer.py",
     "tests/test_ch01_04_foundations.py::test_ch03_bpe_roundtrip_and_compression"),
    ("ch04",    "RMSNorm 与 Pre-Norm",      "norm.py",          "minimind/model/model_minimind.py#L50-L60",
     "tests/test_ch01_04_foundations.py::test_ch04_rmsnorm_matches_minimind"),
    ("ch05",    "GQA 注意力与 KV Cache",    "attention.py",     "minimind/model/model_minimind.py#L86-L134",
     "tests/test_ch05_07_transformer_blocks.py::test_ch05_attention_and_kv_cache"),
    ("ch06",    "RoPE 旋转位置编码与 YaRN", "rope.py",          "minimind/model/model_minimind.py#L62-L84",
     "tests/test_ch05_07_transformer_blocks.py::test_ch06_rope_and_yarn_match_minimind"),
    ("ch07",    "SwiGLU 前馈网络",          "ffn.py",           "minimind/model/model_minimind.py#L136-L146",
     "tests/test_ch05_07_transformer_blocks.py::test_ch07_swiglu_ffn"),
    ("ch08-09", "完整模型组装与 MoE",       "model.py,moe.py",  "minimind/model/model_minimind.py#L148-L258",
     "tests/test_ch08_10_model_and_generate.py::test_ch08_09_full_model_and_moe_match_minimind"),
    ("ch10",    "自回归解码与采样策略",     "generate.py",      "minimind/model/model_minimind.py#L261-L293",
     "tests/test_ch08_10_model_and_generate.py::test_ch10_generation_matches_minimind"),
    ("ch11-13", "Pretrain / SFT 数据集与掩码", "data.py",       "minimind/dataset/lm_dataset.py#L40-L122",
     "tests/test_ch11_19_training_and_alignment.py::test_ch11_13_datasets_match_minimind"),
    ("ch12",    "训练循环、LR 调度与梯度累积", "train_utils.py", "minimind/trainer/train_pretrain.py",
     "tests/test_ch11_19_training_and_alignment.py::test_ch12_train_loop_reduces_loss"),
    ("ch14",    "LoRA 参数高效微调与合并",  "lora.py",          "minimind/model/model_lora.py",
     "tests/test_ch11_19_training_and_alignment.py::test_ch14_lora_zero_init_and_merge"),
    ("ch15",    "白盒知识蒸馏 (CE + KL)",   "distill.py",       "minimind/trainer/train_distillation.py",
     "tests/test_ch11_19_training_and_alignment.py::test_ch15_distillation_loss"),
    ("ch16",    "DPO 直接偏好优化",         "dpo.py",           "minimind/trainer/train_dpo.py",
     "tests/test_ch11_19_training_and_alignment.py::test_ch16_dpo_increases_chosen_margin"),
    ("ch17-18", "RLAIF: PPO (GAE) 与 GRPO/CISPO", "ppo.py,grpo.py", "minimind/trainer/train_ppo.py,train_grpo.py",
     "tests/test_ch11_19_training_and_alignment.py::test_ch17_18_ppo_and_grpo"),
    ("ch19",    "Tool Calling 与 Agentic RL", "agent.py",       "minimind/trainer/train_agent.py",
     "tests/test_ch11_19_training_and_alignment.py::test_ch19_agent_tool_call"),
]


def count_remaining_todos(pkg_dir: Path, files_csv: str) -> int:
    n = 0
    for fname in files_csv.split(","):
        p = pkg_dir / fname.strip()
        if p.exists():
            n += len(re.findall(r"raise NotImplementedError", p.read_text(encoding="utf-8")))
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("filter", nargs="?", default="", help="只运行匹配的章节，如 ch05")
    ap.add_argument("--ref", action="store_true", help="评测 tinymind 参考实现（默认评测 labs）")
    ap.add_argument("-v", "--verbose", action="store_true", help="打印失败测试的详细报错")
    args = ap.parse_args()

    impl = "tinymind" if args.ref else "labs"
    pkg_dir = ROOT / impl
    if not pkg_dir.exists():
        print(f"未找到 {pkg_dir}，正在运行 tools/make_labs.py 生成练习文件...")
        subprocess.run([sys.executable, str(ROOT / "tools" / "make_labs.py")], check=True)

    env = dict(os.environ, TINYMIND_IMPL=impl)
    selected = [c for c in CHAPTERS if args.filter.lower() in c[0].lower() or args.filter.lower() in c[2].lower()]
    if not selected:
        print(f"没有匹配 '{args.filter}' 的章节。可选：{[c[0] for c in CHAPTERS]}")
        return 1

    # 单次 pytest 进程执行全部选中节点，大幅节省 import torch 时间
    nodes = [c[4] for c in selected]
    res = subprocess.run(
        [sys.executable, "-m", "pytest", "-v", "--tb=short", *nodes],
        cwd=str(ROOT), env=env, capture_output=True, text=True,
    )
    output = res.stdout + res.stderr

    print(f"\n{'=' * 92}")
    print(f"  Learn-MiniMind 闯关评测  |  当前被测目录: {impl}/")
    print(f"{'=' * 92}")
    print(f"{'章节':<9} {'主题':<28} {'练习文件':<18} {'未完成TODO':<10} {'状态'}")
    print(f"{'-' * 92}")

    passed = 0
    for cid, title, files_csv, mm_ref, test_node in selected:
        rem = count_remaining_todos(pkg_dir, files_csv)
        ok = f"{test_node} PASSED" in output
        passed += int(ok)
        status = "✅ PASS" if ok else ("📝 待完成" if rem > 0 else "❌ FAIL")
        print(f"{cid:<9} {title:<28} {files_csv:<18} {rem:<10} {status}")

    print(f"{'-' * 92}")
    print(f"通过: {passed}/{len(selected)} 关   "
          f"(提示: 传入章节号如 `python tools/grade.py ch05 -v` 查看单章详细报错)\n")
    if passed < len(selected) and (args.verbose or len(selected) == 1):
        print(output)
    return 0 if passed == len(selected) else 1


if __name__ == "__main__":
    sys.exit(main())
