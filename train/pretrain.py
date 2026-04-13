import os
import math
import time
import numpy as np
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import swanlab
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
        return (len(self.data) - 1) // self.block_size

    def __getitem__(self, idx):
        start = idx * self.block_size
        chunk = self.data[start : start + self.block_size + 1]
        chunk = torch.from_numpy(chunk.astype(np.int64))
        input_ids = chunk[:-1]
        labels    = chunk[1:]
        return input_ids, labels


# ─────────────────────────────────────────────────────────────
# 2. 学习率调度：warmup + cosine decay（按轮内步数计算）
# ─────────────────────────────────────────────────────────────

def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + cosine * (max_lr - min_lr)


# ─────────────────────────────────────────────────────────────
# 3. 训练一个 step 的逻辑
# ─────────────────────────────────────────────────────────────

def train_step(
    model: MyChatbot,
    optimizer: torch.optim.Optimizer,
    batch_iter,
    device: torch.device,
    accumulation_steps: int,
    scaler: torch.cuda.amp.GradScaler,
) -> float:
    model.train()
    total_loss = 0.0

    for _ in range(accumulation_steps):
        input_ids, labels = next(batch_iter)
        input_ids = input_ids.to(device)
        labels    = labels.to(device)

        with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-1,
            )
            loss = loss / accumulation_steps

        scaler.scale(loss).backward()
        total_loss += loss.item()

    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()

    return total_loss


# ─────────────────────────────────────────────────────────────
# 4. 训练循环（2 epoch × 20000 steps）
# ─────────────────────────────────────────────────────────────

def train():
    # ── 超参数配置 ──────────────────────────────────────────
    block_size         = 512
    batch_size         = 8
    accumulation_steps = 4
    num_epochs         = 2        # ← epoch 数
    steps_per_epoch    = 20000    # ← 每个 epoch 的训练步数
    warmup_steps       = 200
    max_lr             = 3e-4
    min_lr             = max_lr * 0.1
    log_interval       = 50
    save_interval      = 5000
    save_dir           = "./checkpoints"
    bin_path           = "/root/autodl-tmp/MyChatBot/train/data/SpongeBobPRO_pretrain_512_final.bin"

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # ── 模型、优化器 ────────────────────────────────────────
    config = ChatBotConfig()
    model  = MyChatbot(config).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {param_count / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=max_lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')

    # ── SwanLab 初始化 ──────────────────────────────────────
    swanlab.init(
        project="MyChatbot-Pretrain",
        experiment_name="mychatbot-pretrain1",
        config={
            "block_size":         block_size,
            "batch_size":         batch_size,
            "accumulation_steps": accumulation_steps,
            "effective_batch":    batch_size * accumulation_steps,
            "num_epochs":         num_epochs,
            "steps_per_epoch":    steps_per_epoch,
            "total_steps":        num_epochs * steps_per_epoch,
            "warmup_steps":       warmup_steps,
            "max_lr":             max_lr,
            "min_lr":             min_lr,
            "weight_decay":       0.1,
            "betas":              (0.9, 0.95),
            "param_count_M":      round(param_count / 1e6, 2),
            "device":             str(device),
        },
    )

    # ── 数据 ────────────────────────────────────────────────
    dataset    = PretrainDataset(bin_path, block_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    print(f"数据集大小: {len(dataset)} 个样本，"
          f"每 epoch {steps_per_epoch} steps × {accumulation_steps} 累积 = "
          f"{steps_per_epoch * accumulation_steps * batch_size} tokens/epoch")

    # ── 外层 epoch 循环 ────────────────────────────────────
    global_step = 0   # 跨 epoch 的全局步数，用于 SwanLab 横轴对齐

    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{num_epochs} 开始")
        print(f"{'='*50}")

        # 每个 epoch 重新创建迭代器：新的 shuffle 顺序，数据不足时用 cycle 补齐
        from itertools import cycle
        batch_iter = cycle(iter(dataloader))

        t0 = time.time()

        # ── 内层 step 循环 ─────────────────────────────────
        for local_step in range(1, steps_per_epoch + 1):
            global_step += 1

            # ① 学习率：基于轮内步数，每个 epoch 独立走一遍 warmup → cosine
            lr = get_lr(global_step, warmup_steps, num_epochs * steps_per_epoch, max_lr, min_lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # ② 训练一步
            loss = train_step(model, optimizer, batch_iter, device, accumulation_steps, scaler)

            # ③ 打印 + SwanLab 记录
            if local_step % log_interval == 0:
                t1 = time.time()
                dt = t1 - t0
                t0 = t1
                ms_per_step = dt / log_interval * 1000
                print(f"  epoch {epoch} | step {local_step:5d}/{steps_per_epoch} "
                      f"| global {global_step:6d} "
                      f"| loss {loss:.4f} | lr {lr:.2e} | {ms_per_step:.0f}ms/step")

                swanlab.log(
                    {
                        "train/loss":        loss,
                        "train/lr":          lr,
                        "train/ms_per_step": ms_per_step,
                        "train/perplexity":  math.exp(min(loss, 20)),
                        "train/epoch":       epoch,   # 方便在 SwanLab 里按 epoch 筛选曲线
                    },
                    step=global_step,   # 横轴用全局步数，两个 epoch 曲线连续不断档
                )

            # ④ 按全局步数定期保存 checkpoint
            if global_step % save_interval == 0:
                ckpt_path = os.path.join(save_dir, f"epoch{epoch}_step{global_step}.pt")
                torch.save({
                    'epoch':       epoch,
                    'local_step':  local_step,
                    'global_step': global_step,
                    'model':       model.state_dict(),
                    'optimizer':   optimizer.state_dict(),
                    'loss':        loss,
                    'config':      config,
                }, ckpt_path)
                print(f"  ✅ checkpoint 保存至 {ckpt_path}")
                swanlab.log({"checkpoint/saved_at": global_step}, step=global_step)

        # ── epoch 结束：额外保存一次完整 checkpoint ────────
        epoch_ckpt = os.path.join(save_dir, f"epoch{epoch}_final.pt")
        torch.save({
            'epoch':       epoch,
            'global_step': global_step,
            'model':       model.state_dict(),
            'optimizer':   optimizer.state_dict(),
            'loss':        loss,
            'config':      config,
        }, epoch_ckpt)
        print(f"\n✅ Epoch {epoch} 完成，模型保存至 {epoch_ckpt}")
        swanlab.log({"epoch/final_loss": loss, "epoch/index": epoch}, step=global_step)

    print("\n🎉 预训练完成")
    swanlab.finish()


if __name__ == "__main__":
    train()