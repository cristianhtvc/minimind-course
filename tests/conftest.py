"""
pytest 配置：支持通过环境变量 TINYMIND_IMPL=labs 切换被测实现。

- 默认测 tinymind/（参考答案，应该 100% 通过）
- 设置 TINYMIND_IMPL=labs 时测 labs/（你的练习代码）
"""
import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "minimind")):
    if p not in sys.path:
        sys.path.insert(0, p)


def load_module(name: str):
    pkg = os.environ.get("TINYMIND_IMPL", "tinymind")
    return importlib.import_module(f"{pkg}.{name}")
