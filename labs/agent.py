"""
第 19 章：Tool Calling 与 Agentic RL（对应 MiniMind: trainer/train_agent.py + scripts/eval_toolcall.py）

模型通过 chat_template 学会输出：
    <tool_call>
    {"name": "calculate_math", "arguments": {"expression": "256*37"}}
    </tool_call>
外部程序解析 -> 执行工具 -> 把结果包成 <tool_response>...</tool_response> 作为下一轮输入 -> 模型给出最终回答。

Agentic RL 的关键是「奖励设计」：格式是否正确、调用是否合法、最终答案是否包含 Ground Truth……
"""
import ast
import json
import math
import operator
import re
from typing import Dict, List

TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def parse_tool_calls(text: str) -> List[Dict]:
    """从模型输出中解析出所有合法的 JSON 工具调用。"""
    calls = []
    # TODO(ch19-1): 用 TOOL_CALL_RE 找到所有 <tool_call>...</tool_call> 片段，json.loads 解析，失败的跳过
    raise NotImplementedError("ch19-1: 请完成这里的实现，然后运行对应测试")
    return calls


def safe_math_eval(expression: str):
    """只允许算术运算的安全求值（永远不要对模型输出用 eval！）。"""
    bin_ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
               ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
    unary_ops = {ast.UAdd: operator.pos, ast.USub: operator.neg}
    funcs = {n: getattr(math, n) for n in "sqrt exp log log2 log10 sin cos tan floor ceil fabs".split()}

    def walk(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in bin_ops:
            if isinstance(node.op, ast.Pow) and abs(walk(node.right)) > 100:
                raise ValueError("幂运算过大")
            return bin_ops[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in unary_ops:
            return unary_ops[type(node.op)](walk(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in funcs:
            return funcs[node.func.id](*map(walk, node.args))
        raise ValueError(f"不支持的语法: {type(node).__name__}")

    expr = str(expression).replace("^", "**").replace("×", "*").replace("÷", "/").strip()
    if not expr or len(expr) > 256:
        raise ValueError("表达式为空或过长")
    return walk(ast.parse(expr, mode="eval").body)


MOCK_TOOLS = {
    "calculate_math": lambda a: {"result": str(safe_math_eval(a.get("expression", "0")))},
    "get_current_weather": lambda a: {"city": a.get("location"), "temperature": "22°C", "condition": "晴"},
}


def execute_tool(name: str, args: Dict):
    fn = MOCK_TOOLS.get(name)
    if fn is None:
        return {"error": f"未知工具 {name}"}
    try:
        return fn(args)
    except Exception as e:
        return {"error": str(e)}


def validate_gt_in_text(text: str, gt_list: List) -> set:
    """检查最终回答里包含了哪些 Ground Truth（数字按数值比较，其余按子串比较）。"""
    text = str(text)
    nums = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", text.replace(",", ""))]
    hit = set()
    for g in gt_list:
        s = str(g).strip()
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", s.replace(",", "")):
            if any(abs(float(s.replace(",", "")) - n) < 1e-6 for n in nums):
                hit.add(g)
        elif s and s.lower() in text.lower():
            hit.add(g)
    return hit


def tool_reward(turn_outputs: List[str], final_answer: str, gt: List, valid_tool_names: set, unfinished=False) -> float:
    """简化版的 MiniMind Agentic RL 奖励（完整版见 train_agent.py: calculate_rewards）。

    规则：
      - <tool_call> 标签不配对：每处 -0.5
      - 合法调用数与 GT 数一致：+0.5，否则每差一个 -0.5
      - 最终回答命中 GT 的比例 × 2.5
      - 未在最大轮数内完成：-0.5
      - 总分 clip 到 [-3, 3]
    """
    reward = 0.0
    # TODO(ch19-2): 按上面的规则实现奖励函数
    raise NotImplementedError("ch19-2: 请完成这里的实现，然后运行对应测试")
    return reward
