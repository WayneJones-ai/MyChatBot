"""
eval.py
=======
MyChatbot 多轮对话推理脚本。

用法：
    # 加载 SFT 权重对话
    python eval.py --checkpoint train/checkpoints/sft/sft_final.pt

    # 指定生成参数
    python eval.py \
        --checkpoint train/checkpoints/sft/sft_final.pt \
        --max_new_tokens 256 \
        --temperature 0.7 \
        --top_p 0.9 \
        --top_k 50

    # 关闭采样（贪心解码）
    python eval.py --checkpoint train/checkpoints/sft/sft_final.pt --temperature 0.0

对话命令：
    输入内容后回车即可对话
    输入 /clear  → 清空历史，开启新对话
    输入 /history → 查看当前对话历史
    输入 /save   → 将当前对话保存到文件
    输入 /quit 或 /exit 或 Ctrl+C → 退出
"""

import argparse
import json
import os
import sys
import time
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from model.mychatbot import MyChatbot
from model.config import ChatBotConfig
from transformers import PreTrainedTokenizerFast


# ─────────────────────────────────────────────
# 特殊 token（与 sft.py 保持一致）
# ─────────────────────────────────────────────
USER_START   = "<|user|>"
ASSIST_START = "<|assistant|>"
TURN_END     = "<|end|>"


# ─────────────────────────────────────────────
# Tokenizer
# ─────────────────────────────────────────────
def load_tokenizer(tokenizer_dir: str) -> PreTrainedTokenizerFast:
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        tokenizer_dir,
        local_files_only=True,
    )
    print(f"[INFO] Tokenizer 加载完成，词表大小: {tokenizer.vocab_size}")
    return tokenizer


# ─────────────────────────────────────────────
# 模型
# ─────────────────────────────────────────────
def load_model(checkpoint: str, device: torch.device) -> tuple[MyChatbot, ChatBotConfig]:
    print(f"[INFO] 加载权重: {checkpoint}")

    import sys as _sys
    from model import config as _config_module
    _sys.modules.setdefault("config", _config_module)

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)

    # 尝试从 checkpoint 里读取 config
    if isinstance(ckpt, dict) and "config" in ckpt:
        config = ckpt["config"]
        if not isinstance(config, ChatBotConfig):
            # 可能存的是 dict
            config = ChatBotConfig(**config) if isinstance(config, dict) else ChatBotConfig()
    else:
        config = ChatBotConfig()

    model = MyChatbot(config)

    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        print(f"[INFO] epoch={ckpt.get('epoch','N/A')}, loss={ckpt.get('loss','N/A'):.4f}" 
              if isinstance(ckpt.get('loss'), float) else 
              f"[INFO] epoch={ckpt.get('epoch','N/A')}")
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif isinstance(ckpt, dict):
        model.load_state_dict(ckpt)
    else:
        raise ValueError(f"无法识别的 checkpoint 格式: {type(ckpt)}")

    model.to(device)
    model.eval()

    n = sum(p.numel() for p in model.parameters())
    print(f"[INFO] 模型就绪，参数量: {n/1e6:.1f}M，设备: {device}")
    return model, config


# ─────────────────────────────────────────────
# 构建输入序列（多轮历史 + 当前 user 输入）
# ─────────────────────────────────────────────
def build_input_ids(history: list[dict], tokenizer, max_context: int = 512) -> torch.Tensor:
    """
    history 格式：[{"role": "user"|"assistant", "content": "...}, ...]
    最后一条必须是 role="user"。
    拼接后追加 ASSIST_START，让模型续写 assistant 回复。
    若超过 max_context，从最早的轮次开始裁剪（保留至少最近一轮 user）。
    """
    def encode_turn(role: str, content: str) -> list[int]:
        if role == "user":
            return tokenizer.encode(f"{USER_START}{content}{TURN_END}")
        else:
            return tokenizer.encode(f"{ASSIST_START}{content}{TURN_END}")

    # 先把所有历史 token 算出来
    all_ids: list[list[int]] = [encode_turn(t["role"], t["content"]) for t in history]
    assist_prefix = tokenizer.encode(ASSIST_START)

    # 从后往前保留，不超过 max_context
    total = len(assist_prefix)
    keep_from = len(all_ids)
    for i in range(len(all_ids) - 1, -1, -1):
        total += len(all_ids[i])
        if total > max_context:
            keep_from = i + 1
            break
        keep_from = i

    if keep_from > 0:
        print(f"[WARN] 上下文过长，已裁剪最早 {keep_from} 条消息")

    input_ids: list[int] = []
    for ids in all_ids[keep_from:]:
        input_ids.extend(ids)
    input_ids.extend(assist_prefix)   # 提示模型开始输出 assistant

    return torch.tensor([input_ids], dtype=torch.long)


# ─────────────────────────────────────────────
# 生成（支持贪心 / top-k / top-p 采样）
# ─────────────────────────────────────────────
@torch.no_grad()
def generate(
    model: MyChatbot,
    input_ids: torch.Tensor,         # [1, T]
    tokenizer,
    device: torch.device,
    max_new_tokens: int = 256,
    temperature: float  = 0.7,
    top_k: int          = 50,
    top_p: float        = 0.9,
    repetition_penalty: float = 1.1,
) -> str:
    """自回归生成，返回 assistant 回复文本。"""

    end_token_ids: set[int] = set()
    for special in [TURN_END, tokenizer.eos_token]:
        if special:
            ids = tokenizer.encode(special)
            if ids:
                end_token_ids.add(ids[0])

    generated: list[int] = []
    cur_ids = input_ids.to(device)

    for _ in range(max_new_tokens):
        logits = model(cur_ids)          # [1, T, V]
        next_logits = logits[0, -1, :]   # [V]

        # 重复惩罚
        if repetition_penalty != 1.0 and generated:
            for token_id in set(generated):
                if next_logits[token_id] > 0:
                    next_logits[token_id] /= repetition_penalty
                else:
                    next_logits[token_id] *= repetition_penalty

        if temperature == 0.0:
            # 贪心
            next_token = torch.argmax(next_logits).item()
        else:
            next_logits = next_logits / temperature

            # Top-k
            if top_k > 0:
                topk_vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < topk_vals[-1]] = float("-inf")

            # Top-p (nucleus)
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                remove_mask = cum_probs - torch.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove_mask] = float("-inf")
                next_logits = torch.zeros_like(next_logits).scatter_(0, sorted_idx, sorted_logits)

            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

        if next_token in end_token_ids:
            break

        generated.append(next_token)
        cur_ids = torch.cat(
            [cur_ids, torch.tensor([[next_token]], dtype=torch.long, device=device)],
            dim=1,
        )

    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ─────────────────────────────────────────────
# 保存对话到文件
# ─────────────────────────────────────────────
def save_history(history: list[dict], output_dir: str = "eval_logs"):
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"chat_{int(time.time())}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump({"conversations": history}, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 对话已保存至 {filename}")


# ─────────────────────────────────────────────
# 打印对话历史
# ─────────────────────────────────────────────
def print_history(history: list[dict]):
    if not history:
        print("  （对话历史为空）")
        return
    print("─" * 50)
    for i, turn in enumerate(history, 1):
        role = "用户" if turn["role"] == "user" else "助手"
        print(f"  [{i}] {role}: {turn['content']}")
    print("─" * 50)


# ─────────────────────────────────────────────
# 主入口：多轮对话循环
# ─────────────────────────────────────────────
def chat_loop(args):
    # 设备
    if args.device == "auto":
        device = (torch.device("cuda") if torch.cuda.is_available()
                  else torch.device("mps") if torch.backends.mps.is_available()
                  else torch.device("cpu"))
    else:
        device = torch.device(args.device)
    print(f"[INFO] 运行设备: {device}")

    tokenizer = load_tokenizer(args.tokenizer_dir)
    model, _  = load_model(args.checkpoint, device)

    print("\n" + "=" * 60)
    print("  MyChatbot 多轮对话")
    print("  /clear   → 清空历史  /history → 查看历史")
    print("  /save    → 保存对话  /quit    → 退出")
    print("=" * 60 + "\n")

    history: list[dict] = []   # 完整对话历史

    while True:
        try:
            user_input = input("用户: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[INFO] 已退出。")
            break

        if not user_input:
            continue

        # ── 内置命令 ──────────────────────────
        if user_input.lower() in ("/quit", "/exit"):
            print("[INFO] 已退出。")
            break

        if user_input.lower() == "/clear":
            history.clear()
            print("[INFO] 对话历史已清空，开始新对话。\n")
            continue

        if user_input.lower() == "/history":
            print_history(history)
            continue

        if user_input.lower() == "/save":
            save_history(history)
            continue

        # ── 正常对话 ──────────────────────────
        history.append({"role": "user", "content": user_input})

        input_ids = build_input_ids(
            history,
            tokenizer,
            max_context=args.max_context,
        )

        t0 = time.time()
        reply = generate(
            model, input_ids, tokenizer, device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        elapsed = time.time() - t0

        history.append({"role": "assistant", "content": reply})

        print(f"助手: {reply}")
        print(f"      \033[90m[{elapsed:.2f}s | {len(reply)} chars]\033[0m\n")


# ─────────────────────────────────────────────
# 命令行参数
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="MyChatbot 多轮对话推理")

    # 路径
    p.add_argument("--checkpoint",    type=str,
                   default="train/checkpoints/sft/sft_final.pt",
                   help="模型权重路径（.pt）")
    p.add_argument("--tokenizer_dir", type=str,
                   default="tokenizer_15k",
                   help="tokenizer 目录")
    p.add_argument("--device",        type=str, default="auto",
                   choices=["auto", "cpu", "cuda", "mps"])

    # 生成参数
    p.add_argument("--max_new_tokens",     type=int,   default=256,
                   help="最多生成 token 数")
    p.add_argument("--max_context",        type=int,   default=512,
                   help="输入上下文最大长度（超出则裁剪旧轮次）")
    p.add_argument("--temperature",        type=float, default=0.7,
                   help="采样温度，0.0 = 贪心")
    p.add_argument("--top_k",             type=int,   default=50,
                   help="Top-k 采样，0 = 不限制")
    p.add_argument("--top_p",             type=float, default=0.9,
                   help="Nucleus 采样概率阈值")
    p.add_argument("--repetition_penalty", type=float, default=1.1,
                   help="重复惩罚系数，1.0 = 不惩罚")

    return p.parse_args()


if __name__ == "__main__":
    chat_loop(parse_args())