"""
pretrain_eval.py
================
专为 MyChatbot (GQA + RoPE + SwiGLU) 模型定制的推理脚本。

目录结构：
    chatbot/
    ├── pretrain_eval.py   ← 本文件
    ├── model/
    │   ├── mychatbot.py   ← MyChatbot / ChatBotBlock 等
    │   └── config.py      ← ChatBotConfig
    ├── tokenizer_15k/
    │   ├── tokenizer.json
    │   ├── tokenizer_config.json
    │   ├── vocab.json
    │   └── merges.txt
    └── train/
        ├── pretrain.py
        └── sft.py

用法：
    # 单次生成（流式输出）
    python pretrain_eval.py --checkpoint model.pt --prompt "今天天气"

    # 交互模式
    python pretrain_eval.py --checkpoint model.pt --interactive

    # 计算 Perplexity
    python pretrain_eval.py --checkpoint model.pt --perplexity --eval_file test.txt
"""

import argparse
import math
import sys
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast

# ── 导入你自己的模型和配置 ─────────────────────────────────────────────────────
from model.mychatbot import MyChatbot       # model.py 里的主模型
from model.config import ChatBotConfig  # config.py 里的配置类


# ===========================================================
# 加载 Tokenizer
# ===========================================================
def load_tokenizer(tokenizer_dir: str) -> PreTrainedTokenizerFast:
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
    print(f"[INFO] Tokenizer 加载完成，词表大小: {tokenizer.vocab_size}")
    if tokenizer.eos_token_id is None:
        print("[WARN] tokenizer 未设置 eos_token，生成将靠 max_new_tokens 截断")
    return tokenizer


# ===========================================================
# 加载模型
# ===========================================================
def load_model(checkpoint_path: str, config: ChatBotConfig, device: torch.device) -> MyChatbot:
    print("[DEBUG] 1. 开始 torch.load ...")
    import sys
    from model import config as _config_module
    sys.modules["config"] = _config_module
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    print("[DEBUG] 2. torch.load 完成")
    
    print("[DEBUG] 3. 开始构建模型结构 ...")
    model = MyChatbot(config)
    print("[DEBUG] 4. 模型构建完成")

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[INFO] epoch={ckpt.get('epoch','N/A')}, loss={ckpt.get('loss','N/A')}")
    elif isinstance(ckpt, dict) and "model" in ckpt:         # ← 新增这个分支
        model.load_state_dict(ckpt["model"])
        print(f"[INFO] epoch={ckpt.get('epoch','N/A')}, loss={ckpt.get('loss','N/A')}")
    elif isinstance(ckpt, dict):
        model.load_state_dict(ckpt)
    elif isinstance(ckpt, MyChatbot):
        model = ckpt
    else:
        raise ValueError(f"无法识别的 checkpoint 格式: {type(ckpt)}")

    print("[DEBUG] 6. 移动模型到设备 ...")
    model.to(device)
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"[INFO] 模型就绪，参数量: {n/1e6:.1f}M，设备: {device}")
    return model

# ===========================================================
# 单次推理（流式 or 批量）
# ===========================================================
@torch.no_grad()
def run_inference(
    model: MyChatbot,
    tokenizer: PreTrainedTokenizerFast,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    device: torch.device = torch.device("cpu"),
    stream: bool = True,
) -> str:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    eos_id = tokenizer.eos_token_id

    print(f"\n{'─'*55}")
    print(f" Prompt : {prompt}")
    print(f"{'─'*55}")

    if stream:
        # 逐 token 打印
        print(" 生成  : ", end="", flush=True)
        kv_caches = None
        generated_ids = []

        for _ in range(max_new_tokens):
            cur_input = input_ids if kv_caches is None else input_ids[:, -1:]
            logits, kv_caches = model.forward(cur_input, kv_caches=kv_caches, use_cache=True)

            next_logits = logits[:, -1, :]
            if temperature > 0:
                next_logits = next_logits / temperature
            if top_k > 0:
                topk_vals, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < topk_vals[:, -1:]] = float("-inf")

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_id = next_token.item()
            generated_ids.append(token_id)

            token_str = tokenizer.decode([token_id], skip_special_tokens=True)
            print(token_str, end="", flush=True)

            input_ids = torch.cat([input_ids, next_token], dim=1)
            if eos_id is not None and token_id == eos_id:
                break

        print(f"\n{'─'*55}\n")
        result = tokenizer.decode(generated_ids, skip_special_tokens=True)

    else:
        # 使用模型内置 generate（一次性）
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=eos_id,
        )
        new_ids = output_ids[0, input_ids.shape[1]:].tolist()
        result = tokenizer.decode(new_ids, skip_special_tokens=True)
        print(f" 生成  : {result}")
        print(f"{'─'*55}\n")

    return result


# ===========================================================
# 交互模式
# ===========================================================
def interactive_mode(model, tokenizer, args, device):
    print("\n🤖 交互模式（exit 退出 | !set temp=0.9 / topk=40 / max=300 调参）\n")
    temperature    = args.temperature
    top_k          = args.top_k
    max_new_tokens = args.max_new_tokens

    while True:
        try:
            user_input = input(">>> ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("再见！")
                break

            # 参数调整命令
            if user_input.startswith("!set "):
                try:
                    key, val = user_input[5:].split("=")
                    key = key.strip()
                    if key == "temp":
                        temperature = float(val); print(f"[设置] temperature={temperature}")
                    elif key == "topk":
                        top_k = int(val);          print(f"[设置] top_k={top_k}")
                    elif key == "max":
                        max_new_tokens = int(val); print(f"[设置] max_new_tokens={max_new_tokens}")
                    else:
                        print("[提示] 可设置: temp / topk / max")
                except Exception:
                    print("[提示] 格式: !set temp=0.8")
                continue

            run_inference(
                model, tokenizer,
                prompt=user_input,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                device=device,
                stream=True,
            )
        except KeyboardInterrupt:
            print("\n退出")
            break


# ===========================================================
# Perplexity 评估
# ===========================================================
@torch.no_grad()
def compute_perplexity(
    model: MyChatbot,
    tokenizer: PreTrainedTokenizerFast,
    texts: list[str],
    device: torch.device,
    block_size: int = 512,
) -> float:
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    total_loss, total_tokens = 0.0, 0

    for i, text in enumerate(texts):
        ids = tokenizer.encode(text)
        if len(ids) < 2:
            continue
        ids_t = torch.tensor([ids], dtype=torch.long, device=device)

        for start in range(0, len(ids) - 1, block_size):
            chunk = ids_t[:, start: start + block_size + 1]
            if chunk.size(1) < 2:
                continue
            x, y = chunk[:, :-1], chunk[:, 1:]
            logits = model(x)   # [1, T, vocab]
            total_loss   += criterion(logits.view(-1, logits.size(-1)), y.view(-1)).item()
            total_tokens += y.numel()

        if (i + 1) % 50 == 0:
            print(f"  已处理 {i+1}/{len(texts)} 条...")

    return math.exp(total_loss / total_tokens) if total_tokens else float("inf")


# ===========================================================
# 命令行参数
# ===========================================================
def parse_args():
    p = argparse.ArgumentParser(description="MyChatbot 推理 / 评估脚本")

    # 路径
    p.add_argument("--checkpoint",      type=str, required=True)
    p.add_argument("--tokenizer_dir",   type=str, default="tokenizer_15k")
    p.add_argument("--device",          type=str, default="auto",
                   choices=["auto", "cpu", "cuda", "mps"])

    # 生成参数
    p.add_argument("--prompt",          type=str, default="今天天气")
    p.add_argument("--max_new_tokens",  type=int, default=200)
    p.add_argument("--temperature",     type=float, default=0.8)
    p.add_argument("--top_k",           type=int, default=50)
    p.add_argument("--no_stream",       action="store_true",
                   help="关闭流式输出，一次性打印结果")

    # 模式
    p.add_argument("--interactive",     action="store_true")
    p.add_argument("--perplexity",      action="store_true")
    p.add_argument("--eval_file",       type=str, default=None)

    # 模型超参（可覆盖 config.py 默认值）
    p.add_argument("--hidden_size",          type=int, default=None)
    p.add_argument("--num_hidden_layers",    type=int, default=None)
    p.add_argument("--num_attention_heads",  type=int, default=None)
    p.add_argument("--num_key_value_heads",  type=int, default=None)
    p.add_argument("--intermediate_size",    type=int, default=None)
    p.add_argument("--vocab_size",           type=int, default=None)

    return p.parse_args()


# ===========================================================
# 主程序
# ===========================================================
def main():
    args = parse_args()

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

    # Config（命令行可覆盖）
    kwargs = {k: getattr(args, k)
              for k in ("hidden_size", "num_hidden_layers", "num_attention_heads",
                        "num_key_value_heads", "intermediate_size")
              if getattr(args, k) is not None}
    kwargs["vocab_size"] = args.vocab_size or tokenizer.vocab_size
    config = ChatBotConfig(**kwargs)

    # 模型
    model = load_model(args.checkpoint, config, device)

    # 模式
    if args.perplexity:
        if not args.eval_file:
            sys.exit("[ERROR] 请通过 --eval_file 指定评估文本文件")
        with open(args.eval_file, "r", encoding="utf-8") as f:
            texts = [l.strip() for l in f if l.strip()]
        print(f"[INFO] 计算 {len(texts)} 条文本的 Perplexity ...")
        ppl = compute_perplexity(model, tokenizer, texts, device)
        print(f"\n[RESULT] Perplexity = {ppl:.4f}")

    elif args.interactive:
        interactive_mode(model, tokenizer, args, device)

    else:
        run_inference(
            model, tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device,
            stream=not args.no_stream,
        )


if __name__ == "__main__":
    main()