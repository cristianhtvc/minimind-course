"""
第 14 章：LoRA 低秩适配（对应 MiniMind: model/model_lora.py）

W' = W + ΔW = W + B @ A * (alpha / r)
  - W: [out, in] 冻结不动
  - A: [r, in]  高斯初始化；B: [out, r]  零初始化 -> 训练开始时 ΔW = 0，模型行为与原模型完全一致
  - 可训练参数从 out*in 降到 r*(in+out)
"""
import torch
from torch import nn


class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank, lora_alpha=None):
        super().__init__()
        self.rank = rank
        scaling = (rank if lora_alpha is None else lora_alpha) / rank
        self.register_buffer("scaling", torch.tensor(scaling, dtype=torch.float32))
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        # TODO(ch14-1): A 用 N(0, 0.02) 初始化，B 全零初始化
        # >>> SOLUTION
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        self.B.weight.data.zero_()
        # <<< SOLUTION

    def forward(self, x):
        # TODO(ch14-2): 返回 B(A(x)) * scaling
        # >>> SOLUTION
        return self.B(self.A(x)) * self.scaling
        # <<< SOLUTION


def apply_lora(model, rank=16, lora_alpha=None, target=None):
    """给模型中的 Linear 层挂上 LoRA 旁路（通过替换 module.forward 实现，不改动原有参数名）。

    target(name, module) -> bool：选择哪些层；默认与 MiniMind 一致，只选 in_features == out_features 的方阵层
    （在 MiniMind 默认配置下就是 q_proj 和 o_proj）。
    """
    target = target or (lambda name, m: m.in_features == m.out_features)
    device = next(model.parameters()).device
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and target(name, module):
            lora = LoRA(module.in_features, module.out_features, rank=rank, lora_alpha=lora_alpha).to(device)
            setattr(module, "lora", lora)
            original_forward = module.forward

            # 用默认参数「显式绑定」当前循环的变量，避免 Python 闭包的晚绑定陷阱
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)

            module.forward = forward_with_lora
    return model


def mark_only_lora_trainable(model):
    """冻结所有非 LoRA 参数，返回 LoRA 参数列表（交给优化器）。"""
    lora_params = []
    for name, p in model.named_parameters():
        p.requires_grad = "lora" in name
        if p.requires_grad:
            lora_params.append(p)
    return lora_params


def lora_state_dict(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if ".lora." in k}


@torch.no_grad()
def merge_lora(model):
    """把 LoRA 增量合并回原始权重，返回一个不含 lora 键的 state_dict（推理时零额外开销）。"""
    state_dict = {k: v.clone() for k, v in model.state_dict().items() if ".lora." not in k}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and hasattr(module, "lora"):
            # TODO(ch14-3): W_merged = W + (B.weight @ A.weight) * scaling
            # >>> SOLUTION
            delta = (module.lora.B.weight.float() @ module.lora.A.weight.float()) * module.lora.scaling.float()
            state_dict[f"{name}.weight"] = (module.weight.float() + delta).to(module.weight.dtype)
            # <<< SOLUTION
    return state_dict
