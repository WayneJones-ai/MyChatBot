import os
import math
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model'))
from mychatbot import MyChatbot
from config import ChatBotConfig


# ─────────────────────────────────────────────────────────────
# 1. 数据加载
# ─────────────────────────────────────────────────────────────

class PretrainDataset(Dataset):
    def __init__(self, bin_path: str, block_size: int):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        self.block_size = block_size

    def __len__(self):
        # 每个样本占 block_size+1 个 token（多1个用来错位做label）
        return (len(self.data) - 1) // self.block_size

    def __getitem__(self, idx):
        start = idx * self.block_size
        chunk = self.data[start : start + self.block_size + 1]
        chunk = torch.from_numpy(chunk.astype(np.int64))
        input_ids = chunk[:-1]   # [0, block_size)
        labels    = chunk[1:]    # [1, block_size+1)  ← 错位一格
        return input_ids, labels


# ─────────────────────────────────────────────────────────────
# 2. 学习率调度：warmup + cosine decay
# ─────────────────────────────────────────────────────────────

def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    # warmup 阶段：线性从 0 升到 max_lr
    if step < warmup_steps:
        return max_lr * step / warmup_steps

    # 训练结束后：保持最小学习率
    if step >= max_steps:
        return min_lr

    # cosine decay 阶段：从 max_lr 平滑降到 min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)  # 0 → 1
    cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))          # 1 → 0
    return min_lr + cosine * (max_lr - min_lr)


# ─────────────────────────────────────────────────────────────
# 3. 训练一个 step 的逻辑
# ─────────────────────────────────────────────────────────────

def train_step(
    model: MyChatbot,
    optimizer: torch.optim.Optimizer,
    batch_iter,             # dataloader 的迭代器
    device: torch.device,
    accumulation_steps: int,
    scaler: torch.cuda.amp.GradScaler,
) -> float:
    model.train()
    total_loss = 0.0

    for micro_step in range(accumulation_steps):
        input_ids, labels = next(batch_iter)
        input_ids = input_ids.to(device)
        labels    = labels.to(device)

        # 混合精度（AMP）：用 float16 做前向，节省显存、加快速度
        # 为什么不全程 float16？因为 float16 精度低，累积梯度会有误差，
        # 所以梯度更新仍用 float32，只有前向用 float16。
        with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
            logits = model(input_ids)           # [B, seq, vocab_size]

            # cross entropy 要求:
            #   input: [N, vocab_size]  ← 把 batch 和 seq 维合并
            #   target: [N]
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-1,    # -1 位置不计入损失（padding 用）
            )
            # 除以 accumulation_steps：让累积后的梯度等价于单次大 batch 的平均
            loss = loss / accumulation_steps

        scaler.scale(loss).backward()
        total_loss += loss.item()

    # 梯度裁剪：防止梯度爆炸
    # 当梯度的 L2 范数超过 max_norm，就等比例缩小所有梯度
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()

    return total_loss


# ─────────────────────────────────────────────────────────────
# 4. 训练循环
#
# 这里把所有模块串起来，逻辑是：
#   for step in range(max_steps):
#       ① 调整学习率
#       ② 跑一个 train_step（含梯度累积）
#       ③ 每隔一段打印日志
#       ④ 每隔一段保存 checkpoint
# ─────────────────────────────────────────────────────────────

def train():
    # ── 超参数配置 ──────────────────────────────────────────
    block_size         = 512
    batch_size         = 8      # 每个 micro step 的 batch 大小
    accumulation_steps = 4      # 梯度累积步数，等效 batch = 8*4 = 32
    max_steps          = 10000
    warmup_steps       = 200
    max_lr             = 3e-4
    min_lr             = max_lr * 0.1
    log_interval       = 50     # 每 50 step 打印一次
    save_interval      = 500    # 每 500 step 保存一次
    save_dir           = "./checkpoints"
    bin_path           = "train\data\SpongeBobPRO_pretrain_512_final.bin"

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # ── 模型、优化器 ────────────────────────────────────────
    config = ChatBotConfig()
    model  = MyChatbot(config).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # AdamW：Adam + 权重衰减
    # 权重衰减是正则化手段，防止参数过大导致过拟合
    # betas=(0.9, 0.95) 是 LLaMA 的常用设置，比默认 (0.9, 0.999) 更适合语言模型
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=max_lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    # GradScaler 配合 AMP 使用：自动处理 float16 下的梯度缩放
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')

    # ── 数据 ────────────────────────────────────────────────
    dataset    = PretrainDataset(bin_path, block_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    # 用 cycle 让 dataloader 无限循环，避免 step 数超过数据集长度时报错
    from itertools import cycle
    batch_iter = cycle(iter(dataloader))

    # ── 训练循环 ────────────────────────────────────────────
    t0 = time.time()
    for step in range(1, max_steps + 1):

        # ① 调整学习率：每一步都更新
        lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # ② 训练一步（含梯度累积）
        loss = train_step(model, optimizer, batch_iter, device, accumulation_steps, scaler)

        # ③ 打印日志
        if step % log_interval == 0:
            t1    = time.time()
            dt    = t1 - t0
            t0    = t1
            print(f"step {step:5d} | loss {loss:.4f} | lr {lr:.2e} | {dt/log_interval*1000:.0f}ms/step")

        # ④ 保存 checkpoint
        if step % save_interval == 0:
            ckpt_path = os.path.join(save_dir, f"step_{step}.pt")
            torch.save({
                'step':       step,
                'model':      model.state_dict(),
                'optimizer':  optimizer.state_dict(),
                'loss':       loss,
                'config':     config,
            }, ckpt_path)
            print(f"✅ checkpoint 保存至 {ckpt_path}")

    print("🎉 预训练完成")


if __name__ == "__main__":
    train()