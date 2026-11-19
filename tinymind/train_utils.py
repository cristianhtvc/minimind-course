"""
第 12 章：训练循环工程（对应 MiniMind: trainer/trainer_utils.py + trainer/train_pretrain.py）

一个工业级 LLM 训练循环需要的要素：
  - 学习率调度（MiniMind：余弦退火到 10%）
  - 混合精度（bf16 autocast；fp16 时还需要 GradScaler）
  - 梯度累积（用小 batch 模拟大 batch）
  - 梯度裁剪（防止梯度爆炸）
  - 断点续训（保存 model + optimizer + step）
  - 分布式（DDP，见文档，这里不展开）
"""
import math
import os
import random
import time
from contextlib import nullcontext

import numpy as np
import torch


def get_lr(current_step: int, total_steps: int, lr: float) -> float:
    """MiniMind 的学习率调度：从 lr 余弦衰减到 0.1*lr。

    lr(t) = lr * (0.1 + 0.45 * (1 + cos(pi * t / T)))
      t=0  -> lr * (0.1 + 0.9) = lr
      t=T  -> lr * 0.1
    """
    # TODO(ch12-1): 实现上面的公式
    # >>> SOLUTION
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))
    # <<< SOLUTION


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer: str = None) -> str:
    if prefer:
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_autocast(device: str, dtype: str = "bfloat16"):
    """CPU 上不开混合精度；CUDA 上用 bf16/fp16 autocast。"""
    if "cuda" not in device:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16 if dtype == "bfloat16" else torch.float16)


def train_epoch(model, loader, optimizer, *, epoch=0, epochs=1, lr=5e-4, accumulation_steps=1, grad_clip=1.0,
                device="cpu", autocast_ctx=None, scaler=None, log_interval=10, loss_fn=None, logger=print):
    """通用的单个 epoch 训练函数（预训练 / SFT 都可以用）。

    loss_fn(model, batch) -> (loss, info_dict)；默认是 Causal LM 的 (input_ids, labels) 损失 + MoE aux loss。
    """
    autocast_ctx = autocast_ctx or nullcontext()
    iters = len(loader)
    model.train()
    start = time.time()
    history = []

    def default_loss_fn(model, batch):
        input_ids, labels = (t.to(device) for t in batch)
        res = model(input_ids, labels=labels)
        return res.loss + res.aux_loss, {"aux_loss": res.aux_loss.item()}

    loss_fn = loss_fn or default_loss_fn
    for step, batch in enumerate(loader, start=1):
        cur_lr = get_lr(epoch * iters + step, epochs * iters, lr)
        for group in optimizer.param_groups:
            group["lr"] = cur_lr

        with autocast_ctx:
            loss, info = loss_fn(model, batch)
            # TODO(ch12-2): 梯度累积 —— 损失要除以 accumulation_steps（为什么？见第 12 章）
            # >>> SOLUTION
            loss = loss / accumulation_steps
            # <<< SOLUTION

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # TODO(ch12-3): 每 accumulation_steps 步（或最后一步）：
        #   (scaler.unscale_) -> 梯度裁剪 clip_grad_norm_ -> optimizer.step (scaler.step/update) -> zero_grad
        # >>> SOLUTION
        if step % accumulation_steps == 0 or step == iters:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        # <<< SOLUTION

        real_loss = loss.item() * accumulation_steps
        history.append(real_loss)
        if log_interval and (step % log_interval == 0 or step == iters):
            extra = " ".join(f"{k}: {v:.4f}" for k, v in info.items())
            logger(f"Epoch [{epoch + 1}/{epochs}] ({step}/{iters}) loss: {real_loss:.4f} {extra} "
                   f"lr: {cur_lr:.2e} elapsed: {time.time() - start:.1f}s")
    return history


def save_checkpoint(path, model, optimizer=None, **extra):
    """断点续训：同时保存 model / optimizer / 训练进度。先写 .tmp 再原子替换，防止写一半断电。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = {"model": model.state_dict(), **extra}
    if optimizer is not None:
        data["optimizer"] = optimizer.state_dict()
    torch.save(data, path + ".tmp")
    os.replace(path + ".tmp", path)


def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    data = torch.load(path, map_location=map_location)
    model.load_state_dict(data["model"])
    if optimizer is not None and "optimizer" in data:
        optimizer.load_state_dict(data["optimizer"])
    return data
