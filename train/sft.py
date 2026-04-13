"""
sft.py
======
针对 MyChatbot 的监督微调（SFT）训练脚本。

数据格式（jsonl，每行一条）：
    {"conversations": [
        {"role": "user",      "content": "..."},
        {"role": "assistant", "content": "..."},
        {"role": "user",      "content": "..."},  # 多轮可选
        {"role": "assistant", "content": "..."}
    ]}

目录结构：
    chatbot/
    ├── train/
    │   └── sft.py          ← 本文件
    ├── model/
    │   ├── mychatbot.py
    │   └── config.py
    └── tokenizer_15k/

用法：
    # 从头 SFT（无预训练权重）
    python train/sft.py --data_file train/data/sft.jsonl

    # 在预训练权重上继续 SFT
    python train/sft.py \
        --data_file  train/data/sft.jsonl \
        --checkpoint train/checkpoints/epoch2_final.pt \
        --output_dir train/checkpoints/sft

    # 常用参数
    python train/sft.py \
        --data_file  train/data/sft.jsonl \
        --checkpoint train/checkpoints/epoch2_final.pt \
        --epochs 3 --batch_size 4 --lr 2e-5 --max_len 512
"""

import argparse
import json
import math
import os
import sys
import time
import swanlab
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast

# 把项目根目录加入 sys.path，保证从 train/ 下运行也能 import model
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from model.mychatbot import MyChatbot
from model.config import ChatBotConfig


# ─────────────────────────────────────────────
# 特殊 token 模板（可按需修改）
# ─────────────────────────────────────────────
USER_START    = "<|user|>"
ASSIST_START  = "<|assistant|>"
TURN_END      = "<|end|>"


def build_conversation(conversations: list[dict], tokenizer) -> tuple[list[int], list[int]]:
    """
    把多轮对话拼成一个 token 序列，并返回对应的 label 序列。
    只对 assistant 回复部分计算 loss（user 部分 label = -100）。

    返回:
        input_ids : list[int]
        labels    : list[int]   (-100 表示忽略)
    """
    input_ids: list[int] = []
    labels:    list[int] = []

    for turn in conversations:
        role    = turn["role"]
        content = turn["content"]

        if role == "user":
            text = f"{USER_START}{content}{TURN_END}"
            ids  = tokenizer.encode(text)
            input_ids.extend(ids)
            labels.extend([-100] * len(ids))          # user 部分不参与 loss

        elif role == "assistant":
            prefix_ids  = tokenizer.encode(ASSIST_START)
            content_ids = tokenizer.encode(content + TURN_END)

            input_ids.extend(prefix_ids)
            labels.extend([-100] * len(prefix_ids))   # <|assistant|> 标记本身不计 loss

            input_ids.extend(content_ids)
            labels.extend(content_ids)                # assistant 回复计算 loss

    return input_ids, labels


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class SFTDataset(Dataset):
    def __init__(self, data_file: str, tokenizer, max_len: int = 512):
        self.samples = []
        skipped = 0

        print("[INFO] 开始读取数据文件...")          # ← 加
        with open(data_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        print(f"[INFO] 文件读取完成，共 {len(lines)} 行，开始 tokenize...")  # ← 加

        for line_no, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] 第 {line_no} 行 JSON 解析失败，跳过")
                skipped += 1
                continue

            conversations = obj.get("conversations", [])
            if not conversations:
                skipped += 1
                continue

            ids, lbls = build_conversation(conversations, tokenizer)

            if len(ids) > max_len:
                ids  = ids[:max_len]
                lbls = lbls[:max_len]

            if all(l == -100 for l in lbls):
                skipped += 1
                continue

            self.samples.append((ids, lbls))

            # ← 每 1000 条打印一次进度
            if line_no % 1000 == 0:
                print(f"[INFO] 已处理 {line_no}/{len(lines)} 条...")

        print(f"[INFO] 数据集加载完成：{len(self.samples)} 条有效，{skipped} 条跳过")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch, pad_id: int = 0):
    """动态 padding 到 batch 内最长序列。"""
    max_len = max(len(ids) for ids, _ in batch)

    input_batch  = []
    label_batch  = []
    attn_batch   = []

    for ids, lbls in batch:
        pad_len = max_len - len(ids)

        input_batch.append(ids  + [pad_id]  * pad_len)
        label_batch.append(lbls + [-100]    * pad_len)   # padding 位置不计 loss
        attn_batch.append([1] * len(ids) + [0] * pad_len)

    return (
        torch.tensor(input_batch,  dtype=torch.long),
        torch.tensor(label_batch,  dtype=torch.long),
        torch.tensor(attn_batch,   dtype=torch.long),
    )


# ─────────────────────────────────────────────
# 加载 Tokenizer
# ─────────────────────────────────────────────
def load_tokenizer(tokenizer_dir: str) -> PreTrainedTokenizerFast:
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        tokenizer_dir,
        local_files_only=True,
    )
    print(f"[INFO] Tokenizer 加载完成，词表大小: {tokenizer.vocab_size}")
    return tokenizer


# ─────────────────────────────────────────────
# 加载模型（兼容多种 checkpoint 格式）
# ─────────────────────────────────────────────
def load_model(config: ChatBotConfig, checkpoint: str | None, device: torch.device) -> MyChatbot:
    model = MyChatbot(config)

    if checkpoint:
        print(f"[INFO] 加载预训练权重: {checkpoint}")

        # 兼容旧 checkpoint（config 保存在根目录时）
        import sys as _sys
        from model import config as _config_module
        _sys.modules.setdefault("config", _config_module)

        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)

        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"[INFO] 预训练 epoch={ckpt.get('epoch','N/A')}, loss={ckpt.get('loss','N/A')}")
        elif isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
            print(f"[INFO] 预训练 epoch={ckpt.get('epoch','N/A')}, loss={ckpt.get('loss','N/A')}")
        elif isinstance(ckpt, dict):
            model.load_state_dict(ckpt)
        else:
            raise ValueError(f"无法识别的 checkpoint 格式: {type(ckpt)}")
    else:
        print("[INFO] 未指定 checkpoint，从随机初始化开始 SFT")

    model.to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f"[INFO] 模型就绪，参数量: {n/1e6:.1f}M，设备: {device}")
    return model


# ─────────────────────────────────────────────
# 保存 checkpoint
# ─────────────────────────────────────────────
def save_checkpoint(model, optimizer, epoch, step, loss, config, output_dir, tag):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{tag}.pt")
    torch.save(
        {
            "epoch":        epoch,
            "global_step":  step,
            "model":        model.state_dict(),
            "optimizer":    optimizer.state_dict(),
            "loss":         loss,
            "config":       config,
        },
        path,
    )
    print(f"[SAVE] checkpoint → {path}")


# ─────────────────────────────────────────────
# 训练主循环
# ─────────────────────────────────────────────
def train(args):
    # 设备
    if args.device == "auto":
        device = (torch.device("cuda") if torch.cuda.is_available()
                  else torch.device("mps") if torch.backends.mps.is_available()
                  else torch.device("cpu"))
    else:
        device = torch.device(args.device)
    print(f"[INFO] 运行设备: {device}")

    # Tokenizer
    tokenizer = load_tokenizer(args.tokenizer_dir)
    pad_id = tokenizer.pad_token_id or 0

    # Dataset & DataLoader
    dataset = SFTDataset(args.data_file, tokenizer, max_len=args.max_len)
    loader  = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_id=pad_id),
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Config & 模型
    config = ChatBotConfig(vocab_size=tokenizer.vocab_size)
    model  = load_model(config, args.checkpoint, device)
    model.train()

    # 优化器 & 调度器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    swanlab.init(
        project="MyChatbot-SFT",          # 项目名，可自定义
        experiment_name=f"sft-lr{args.lr}-bs{args.batch_size}",
        config={
            "epochs":        args.epochs,
            "batch_size":    args.batch_size,
            "max_len":       args.max_len,
            "lr":            args.lr,
            "weight_decay":  args.weight_decay,
            "grad_clip":     args.grad_clip,
            "warmup_ratio":  args.warmup_ratio,
            "checkpoint":    args.checkpoint,
            "vocab_size":    tokenizer.vocab_size,
        },
    )

    total_steps = len(loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(args.min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 损失函数
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # ── 训练循环 ──────────────────────────────
    global_step = 0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        epoch_loss  = 0.0
        epoch_start = time.time()

        for step, (input_ids, labels, _attention_mask) in enumerate(loader, 1):
            input_ids = input_ids.to(device)
            labels    = labels.to(device)

            # 前向：取 input 除最后一位，预测 label 除第一位
            # 或直接传整个序列让模型自己对齐（视 MyChatbot.forward 实现而定）
            logits = model(input_ids)          # [B, T, V]

            # logits 与 labels 对齐：预测下一个 token
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = criterion(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            global_step += 1
            epoch_loss  += loss.item()

            # 日志
            if step % args.log_interval == 0:
                avg = epoch_loss / step
                lr  = scheduler.get_last_lr()[0]
                elapsed = time.time() - epoch_start
                print(
                    f"Epoch {epoch}/{args.epochs} | "
                    f"Step {step}/{len(loader)} | "
                    f"Loss {loss.item():.4f} | "
                    f"Avg {avg:.4f} | "
                    f"LR {lr:.2e} | "
                    f"{elapsed:.0f}s"
                )
                swanlab.log({
                    "train/loss":     loss.item(),
                    "train/avg_loss": avg,
                    "train/lr":       lr,
                    "train/epoch":    epoch,
                }, step=global_step)

            # 中间保存
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                save_checkpoint(
                    model, optimizer, epoch, global_step,
                    loss.item(), config, args.output_dir,
                    tag=f"step{global_step}",
                )

        # Epoch 结束保存
        avg_loss = epoch_loss / len(loader)
        elapsed  = time.time() - epoch_start
        print(f"\n{'='*60}")
        print(f"Epoch {epoch} 完成 | 平均 Loss: {avg_loss:.4f} | 耗时: {elapsed:.0f}s")
        print(f"{'='*60}\n")
        swanlab.log({
            "epoch/avg_loss": avg_loss,
            "epoch/epoch":    epoch,
        }, step=global_step)
        save_checkpoint(
            model, optimizer, epoch, global_step,
            avg_loss, config, args.output_dir,
            tag=f"sft_epoch{epoch}",
        )

    # 最终保存
    save_checkpoint(
        model, optimizer, args.epochs, global_step,
        avg_loss, config, args.output_dir,
        tag="sft_final",
    )
    print("[INFO] SFT 训练完成！")
    swanlab.finish()

# ─────────────────────────────────────────────
# 命令行参数
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="MyChatbot SFT 训练脚本")

    # 路径
    p.add_argument("--data_file",     type=str, default="/root/autodl-tmp/MyChatBot/train/data/spongebob_sft.jsonl",help="SFT 数据文件（jsonl）")
    p.add_argument("--checkpoint",    type=str, default="/root/autodl-tmp/MyChatBot/train/checkpoints/epoch2_final.pt",help="预训练权重路径（可选）")
    p.add_argument("--tokenizer_dir", type=str, default="tokenizer_15k")
    p.add_argument("--output_dir",    type=str, default="train/checkpoints/sft")
    p.add_argument("--device",        type=str, default="auto",
                   choices=["auto", "cpu", "cuda", "mps"])

    # 训练超参
    p.add_argument("--epochs",        type=int,   default=2)
    p.add_argument("--batch_size",    type=int,   default=4)
    p.add_argument("--max_len",       type=int,   default=512)
    p.add_argument("--lr",            type=float, default=2e-5)
    p.add_argument("--weight_decay",  type=float, default=0.01)
    p.add_argument("--grad_clip",     type=float, default=1.0)
    p.add_argument("--warmup_ratio",  type=float, default=0.05,         help="warmup 占总步数比例")
    p.add_argument("--min_lr_ratio",  type=float, default=0.1,          help="最小学习率 = lr * min_lr_ratio")

    # 日志与保存
    p.add_argument("--log_interval",  type=int,   default=100,           help="每隔多少 step 打印一次")
    p.add_argument("--save_steps",    type=int,   default=5000,            help="每隔多少 step 保存一次，0=不保存")
    p.add_argument("--num_workers",   type=int,   default=2)

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())