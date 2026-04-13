import torch
import torch.nn as nn
import math
from .config import ChatBotConfig
import torch.nn.functional as F


def precompute_rope(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6):
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin


def rope(q, k, cos, sin, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    # print(f"RoPE:q的维度是{q.shape},k的维度是{k.shape},cos的维度是{cos.shape},sin的维度是{sin.shape}")
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: ChatBotConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        self.q_proj = nn.Linear(self.embed_dim, self.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.embed_dim, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.embed_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        position_inf: tuple[torch.Tensor, torch.Tensor],
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
        attention_mask: torch.Tensor | None = None,
    ):
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)

        q, k = rope(q, k, position_inf[0], position_inf[1])

        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=1)
            v = torch.cat([kv_cache[1], v], dim=1)
        if use_cache:
            kv_cache = (k, v)

        k = repeat_kv(k, self.num_attention_heads // self.num_key_value_heads)
        v = repeat_kv(v, self.num_attention_heads // self.num_key_value_heads)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        q_len, kv_len = scores.shape[-2], scores.shape[-1]
        causal_mask = torch.triu(
            torch.ones(q_len, kv_len, device=x.device),
            diagonal=kv_len - q_len + 1
        ).bool()
        # print(f"GQA:scores的维度是{scores.shape},causal_mask的维度是{causal_mask.shape}")
        scores = scores.masked_fill(causal_mask, float('-inf'))

        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))

        scores = scores.softmax(dim=-1).type_as(scores)
        scores = self.dropout(scores)
        x = torch.matmul(scores, v)
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        x = self.o_proj(x)

        if use_cache:
            return x, kv_cache
        return x


# ── FeedForward：SwiGLU 需要配合线性投影 ─────────────────────────────────────
class FeedForward(nn.Module):
    def __init__(self, config: ChatBotConfig):
        super().__init__()
        # gate 和 up 合并进一个矩阵，减少一次 kernel launch
        self.gate_up_proj = nn.Linear(config.hidden_size, config.intermediate_size * 2, bias=config.bias)
        self.down_proj    = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.bias)
        self.dropout      = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)   # 各 [B, seq, intermediate_size]
        return self.dropout(self.down_proj(F.silu(gate) * up))


# ── ChatBotBlock ──────────────────────────────────────────────────────────────
class ChatBotBlock(nn.Module):
    def __init__(self, config: ChatBotConfig):
        super().__init__()
        # 两个 norm 必须独立，共享同一个对象会导致梯度互相干扰
        self.norm1 = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, elementwise_affine=True)
        self.norm2 = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, elementwise_affine=True)
        self.GQA         = GroupedQueryAttention(config)
        self.FeedForward = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        position_inf: tuple[torch.Tensor, torch.Tensor],
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
        attention_mask: torch.Tensor | None = None,
    ):
        # Attention 子层：Norm + 残差
        if use_cache:
            attn_out, kv_cache = self.GQA(
                self.norm1(x), position_inf, kv_cache, use_cache, attention_mask
            )
        else:
            attn_out = self.GQA(
                self.norm1(x), position_inf, kv_cache, use_cache, attention_mask
            )
        x = x + attn_out

        # FFN 子层：Norm + 残差
        x = x + self.FeedForward(self.norm2(x))

        if use_cache:
            return x, kv_cache
        return x


# ── MyChatbot 主模型 ──────────────────────────────────────────────────────────
class MyChatbot(nn.Module):
    def __init__(self, config: ChatBotConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [ChatBotBlock(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm    = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, elementwise_affine=True)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 权重共享：embedding 与 lm_head 共享，节省显存（LLaMA 同款做法）
        self.lm_head.weight = self.embed_tokens.weight

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,                        # [B, seq]
        kv_caches: list | None = None,                  # 每层一个 (k, v)，推理时传入
        use_cache: bool = False,
        attention_mask: torch.Tensor | None = None,
    ):
        batch_size, seq_len = input_ids.shape
        x = self.embed_tokens(input_ids)                # [B, seq, hidden]

        # RoPE：有 cache 时从 cache 末尾位置开始，保证位置连续
        cache_len = kv_caches[0][0].shape[1] if kv_caches is not None else 0
        freqs_cos, freqs_sin = precompute_rope(
            dim=self.config.hidden_size // self.config.num_attention_heads,
            end=cache_len + seq_len,
            rope_base=self.config.rope_theta,
        )
        freqs_cos = freqs_cos[cache_len:].to(x.device)
        freqs_sin = freqs_sin[cache_len:].to(x.device)
        position_inf = (freqs_cos, freqs_sin)

        new_kv_caches = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            if use_cache:
                x, new_cache = layer(x, position_inf, layer_cache, use_cache, attention_mask)
                new_kv_caches.append(new_cache)
            else:
                x = layer(x, position_inf, layer_cache, use_cache, attention_mask)

        x = self.norm(x)
        logits = self.lm_head(x)                        # [B, seq, vocab_size]

        if use_cache:
            return logits, new_kv_caches
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,                        # [1, prompt_len]
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """自回归生成：首次传完整 prompt，后续每步只传最新一个 token"""
        self.eval()
        kv_caches = None

        for _ in range(max_new_tokens):
            cur_input = input_ids if kv_caches is None else input_ids[:, -1:]
            logits, kv_caches = self.forward(cur_input, kv_caches=kv_caches, use_cache=True)

            next_logits = logits[:, -1, :]

            if temperature != 1.0:
                next_logits = next_logits / temperature
            if top_k > 0:
                topk_vals, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < topk_vals[:, -1:]] = float('-inf')

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [1, 1]
            input_ids = torch.cat([input_ids, next_token], dim=1)

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

        return input_ids