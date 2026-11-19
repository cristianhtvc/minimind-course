"""
从 tinymind/（参考答案）生成 labs/（练习骨架）。

规则：tinymind 源码中被如下标记包裹的代码会被替换为 `raise NotImplementedError(...)`：

        # TODO(ch05-2): 说明……
        # >>> SOLUTION
        ...参考实现...
        # <<< SOLUTION

用法：
    python tools/make_labs.py            # 生成 labs/（已存在的文件不会被覆盖，保护你的作业）
    python tools/make_labs.py --force    # 强制覆盖（会清空你在 labs/ 里写的代码！）
    python tools/make_labs.py --only attention rope   # 只重新生成指定模块
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "tinymind"
DST = ROOT / "labs"

START = re.compile(r"^(\s*)# >>> SOLUTION\s*$")
END = re.compile(r"^\s*# <<< SOLUTION\s*$")
TODO = re.compile(r"TODO\((ch\d+-\d+)\)")


def strip_solutions(text: str, filename: str) -> str:
    out, in_block, last_todo = [], False, None
    for lineno, line in enumerate(text.splitlines(keepends=True), 1):
        m_todo = TODO.search(line)
        if m_todo and not in_block:
            last_todo = m_todo.group(1)
        m_start = START.match(line)
        if m_start:
            if in_block:
                raise SystemExit(f"{filename}:{lineno}: 嵌套的 SOLUTION 块")
            in_block = True
            indent = m_start.group(1)
            tag = last_todo or "TODO"
            out.append(f'{indent}raise NotImplementedError("{tag}: 请完成这里的实现，然后运行对应测试")\n')
            continue
        if END.match(line):
            if not in_block:
                raise SystemExit(f"{filename}:{lineno}: 多余的 <<< SOLUTION")
            in_block = False
            continue
        if not in_block:
            out.append(line)
    if in_block:
        raise SystemExit(f"{filename}: SOLUTION 块未闭合")
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", nargs="*", help="只生成指定模块（不带 .py）")
    args = ap.parse_args()

    DST.mkdir(exist_ok=True)
    n_written, n_skipped, n_todos = 0, 0, 0
    for src in sorted(SRC.glob("*.py")):
        if args.only and src.stem not in args.only:
            continue
        dst = DST / src.name
        text = src.read_text(encoding="utf-8")
        n_todos += len(START.findall(text)) if False else sum(1 for l in text.splitlines() if START.match(l))
        if dst.exists() and not args.force and not args.only:
            n_skipped += 1
            continue
        new = strip_solutions(text, src.name)
        if src.name == "__init__.py":
            new = new.replace("TinyMind —— 跟着 MiniMind 从零手写一个 LLM 的参考实现。",
                              "Labs —— 你的练习区：把每个 NotImplementedError 换成你的实现。")
        dst.write_text(new, encoding="utf-8")
        n_written += 1
    print(f"生成 {n_written} 个文件，跳过 {n_skipped} 个已存在文件（使用 --force 覆盖），共 {n_todos} 个 TODO。")
    print(f"输出目录：{DST}")


if __name__ == "__main__":
    sys.exit(main())
