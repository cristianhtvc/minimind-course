"""
Learn-MiniMind 网页端交互式学习平台后端服务（类似 LeetCode 的本地 PyTorch 评测服务器）。

功能：
  1. 托管交互式课程与在线编译器页面：`learn-minimind/index.html`
  2. 提供实时 PyTorch + MiniMind 对齐评测接口 `POST /api/grade`：
     接收用户在网页编辑器中编写的 Python 代码 -> 自动保存到 `labs/<file>.py` ->
     在真实 PyTorch 环境中运行对应的 pytest 单元测试（与 MiniMind 原版做逐元素数值与梯度比对） ->
     返回每个测试用例的通过状态、耗时与详细报错堆栈。
  3. 提供 `POST /api/pipeline`：在网页端一键运行 30 秒全流程 6 阶段微缩训练实验并返回实时日志。

启动方式：
    ../.venv/bin/python tools/web_ide.py          # 默认监听 http://127.0.0.1:8765 并自动打开浏览器
    ../.venv/bin/python tools/web_ide.py --port 9000 --no-open

注：即使不启动本服务，直接用浏览器双击打开 `learn-minimind/index.html`，
网页内置的离线评测与可视化引擎也完全可用！启动本服务后会自动升级为「真实 PyTorch 内核评测模式」。
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABS_DIR = ROOT / "labs"
TINYMIND_DIR = ROOT / "tinymind"

CHAPTER_META = {
    "ch01-02": {
        "file": "basics.py",
        "test": "tests/test_ch01_04_foundations.py::test_ch01_02_basics",
        "title": "第 1–2 章：数值稳定 Softmax、LogSumExp 交叉熵与困惑度",
    },
    "ch03": {
        "file": "tokenizer_bpe.py",
        "test": "tests/test_ch01_04_foundations.py::test_ch03_bpe_roundtrip_and_compression",
        "title": "第 3 章：从零实现 Byte-Level BPE 分词器",
    },
    "ch04": {
        "file": "norm.py",
        "test": "tests/test_ch01_04_foundations.py::test_ch04_rmsnorm_matches_minimind",
        "title": "第 4 章：RMSNorm 与混合精度防溢出归一化",
    },
    "ch05": {
        "file": "attention.py",
        "test": "tests/test_ch05_07_transformer_blocks.py::test_ch05_attention_and_kv_cache",
        "title": "第 5 章：分组查询注意力 (GQA)、QK-Norm 与 KV Cache",
    },
    "ch06": {
        "file": "rope.py",
        "test": "tests/test_ch05_07_transformer_blocks.py::test_ch06_rope_and_yarn_match_minimind",
        "title": "第 6 章：RoPE 旋转位置编码与 YaRN 长上下文外推",
    },
    "ch07": {
        "file": "ffn.py",
        "test": "tests/test_ch05_07_transformer_blocks.py::test_ch07_swiglu_ffn",
        "title": "第 7 章：SwiGLU 门控前馈网络",
    },
    "ch08": {
        "file": "model.py",
        "test": "tests/test_ch08_10_model_and_generate.py::test_ch08_09_full_model_and_moe_match_minimind",
        "title": "第 8 章：组装 TinyMindForCausalLM（Pre-Norm + Weight Tying + Shift CE）",
    },
    "ch09": {
        "file": "moe.py",
        "test": "tests/test_ch08_10_model_and_generate.py::test_ch08_09_full_model_and_moe_match_minimind",
        "title": "第 9 章：混合专家模型 MoE（Top-K 路由、STE 直通梯度与负载均衡损失）",
    },
    "ch10": {
        "file": "generate.py",
        "test": "tests/test_ch08_10_model_and_generate.py::test_ch10_generation_matches_minimind",
        "title": "第 10 章：自回归解码与五大采样策略（Temperature / RepPenalty / Top-K / Top-P）",
    },
    "ch11-13": {
        "file": "data.py",
        "test": "tests/test_ch11_19_training_and_alignment.py::test_ch11_13_datasets_match_minimind",
        "title": "第 11 & 13 章：PretrainDataset 与 SFTDataset（仅对 Assistant 回复打 Label）",
    },
    "ch12": {
        "file": "train_utils.py",
        "test": "tests/test_ch11_19_training_and_alignment.py::test_ch12_train_loop_reduces_loss",
        "title": "第 12 章：余弦退火学习率、梯度累积与梯度裁剪训练循环",
    },
    "ch14": {
        "file": "lora.py",
        "test": "tests/test_ch11_19_training_and_alignment.py::test_ch14_lora_zero_init_and_merge",
        "title": "第 14 章：手写 LoRA 低秩适配与零开销权重合并",
    },
    "ch15": {
        "file": "distill.py",
        "test": "tests/test_ch11_19_training_and_alignment.py::test_ch15_distillation_loss",
        "title": "第 15 章：白盒知识蒸馏（软标签 KL 散度与 T² 梯度补偿）",
    },
    "ch16": {
        "file": "dpo.py",
        "test": "tests/test_ch11_19_training_and_alignment.py::test_ch16_dpo_increases_chosen_margin",
        "title": "第 16 章：DPO 直接偏好优化（Bradley-Terry 解析隐式奖励）",
    },
    "ch17": {
        "file": "ppo.py",
        "test": "tests/test_ch11_19_training_and_alignment.py::test_ch17_18_ppo_and_grpo",
        "title": "第 17 章：PPO 近端策略优化与 GAE 广义优势估计",
    },
    "ch18": {
        "file": "grpo.py",
        "test": "tests/test_ch11_19_training_and_alignment.py::test_ch17_18_ppo_and_grpo",
        "title": "第 18 章：DeepSeek GRPO 组内相对优势与 MiniMax CISPO 损失",
    },
    "ch19": {
        "file": "agent.py",
        "test": "tests/test_ch11_19_training_and_alignment.py::test_ch19_agent_tool_call",
        "title": "第 19 章：Tool Calling 解析与 Agentic RL 可验证奖励函数 (RLVR)",
    },
}


class WebIDEHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # 保持终端输出清爽

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send_json({"ok": True})

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            html_path = ROOT / "index.html"
            if not html_path.exists():
                self.send_error(404, "index.html not found")
                return
            data = html_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if self.path.startswith("/api/health"):
            import torch
            self._send_json({
                "ok": True,
                "mode": "pytorch-backend",
                "python": sys.version.split()[0],
                "torch": torch.__version__,
            })
            return

        if self.path.startswith("/api/files"):
            files = {}
            for cid, meta in CHAPTER_META.items():
                fname = meta["file"]
                lab_p = LABS_DIR / fname
                ref_p = TINYMIND_DIR / fname
                files[cid] = {
                    "file": fname,
                    "lab_code": lab_p.read_text(encoding="utf-8") if lab_p.exists() else "",
                    "ref_code": ref_p.read_text(encoding="utf-8") if ref_p.exists() else "",
                }
            self._send_json({"ok": True, "files": files})
            return

        self.send_error(404, "Not Found")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        try:
            payload = json.loads(raw)
        except Exception:
            payload = {}

        if self.path == "/api/grade":
            cid = payload.get("chapterId", "")
            code = payload.get("code", "")
            companion_code = payload.get("companionCode")
            if cid not in CHAPTER_META:
                self._send_json({"ok": False, "error": f"未知章节: {cid}"}, 400)
                return

            meta = CHAPTER_META[cid]
            fname = meta["file"]
            LABS_DIR.mkdir(exist_ok=True)
            if code:
                (LABS_DIR / fname).write_text(code, encoding="utf-8")
            # 对于共享同一个 pytest 节点的联合章节（如 ch08/ch09 或 ch17/ch18），若另一文件还有未完成 TODO，临时借用参考实现以隔离评测当前文件
            temp_restores = {}
            paired = {"ch08": "moe.py", "ch09": "model.py", "ch17": "grpo.py", "ch18": "ppo.py"}
            if cid in paired:
                other_file = paired[cid]
                other_lab = LABS_DIR / other_file
                if companion_code:
                    other_lab.write_text(companion_code, encoding="utf-8")
                elif other_lab.exists() and "raise NotImplementedError" in other_lab.read_text(encoding="utf-8"):
                    temp_restores[other_lab] = other_lab.read_text(encoding="utf-8")
                    other_lab.write_text((TINYMIND_DIR / other_file).read_text(encoding="utf-8"), encoding="utf-8")

            t0 = time.time()
            env = dict(os.environ, TINYMIND_IMPL="labs")
            try:
                res = subprocess.run(
                    [sys.executable, "-m", "pytest", "-v", "--tb=short", meta["test"]],
                    cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=30,
                )
                elapsed_ms = int((time.time() - t0) * 1000)
                output = res.stdout + res.stderr
                passed = res.returncode == 0
                rem_todos = len(re.findall(r"raise NotImplementedError", code))
                self._send_json({
                    "ok": True,
                    "passed": passed,
                    "elapsed_ms": elapsed_ms,
                    "remaining_todos": rem_todos,
                    "output": output,
                })
            finally:
                for p, orig_text in temp_restores.items():
                    p.write_text(orig_text, encoding="utf-8")
            return

        if self.path == "/api/pipeline":
            t0 = time.time()
            res = subprocess.run(
                [sys.executable, str(ROOT / "tools" / "train_tiny_pipeline.py")],
                cwd=str(ROOT), capture_output=True, text=True, timeout=60,
            )
            self._send_json({
                "ok": res.returncode == 0,
                "elapsed_ms": int((time.time() - t0) * 1000),
                "output": res.stdout + res.stderr,
            })
            return

        self.send_error(404, "Not Found")


def main():
    ap = argparse.ArgumentParser(description="启动 Learn-MiniMind 网页端交互式学习平台与 LeetCode 评测后端")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-open", action="store_true", help="不自动打开浏览器")
    args = ap.parse_args()

    url = f"http://{args.host}:{args.port}"
    server = HTTPServer((args.host, args.port), WebIDEHandler)
    print(f"\n{'=' * 76}")
    print(f"🚀 Learn-MiniMind 网页端交互式课程 & LeetCode 实时评测平台已启动！")
    print(f"👉 请在浏览器中打开: {url}")
    print(f"{'=' * 76}\n")
    if not args.no_open:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止服务。")


if __name__ == "__main__":
    main()
