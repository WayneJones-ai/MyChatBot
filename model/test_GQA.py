import torch
import torch.nn as nn
import math

# 定义测试用的配置类
class TestConfig:
    def __init__(self):
        # 用于测试的较小参数值
        self.hidden_size = 128       # 原始值可能为 4096
        self.num_attention_heads = 8   # 原始值可能为 32
        self.num_key_value_heads = 4  # 原始值可能为 8
        self.dropout = 0.1
        self.max_position_embeddings = 1024
        self.rope_theta = 10000.0

# 从mychatbot模块导入必要的组件
from mychatbot import (
    precompute_rope,
    rope,
    repeat_kv,
    GroupedQueryAttention
)

# 构造输入
batch_size = 2
seq_len = 10

def test_forward():
    """测试前向传播功能"""
    config = TestConfig()
    model = GroupedQueryAttention(config)
    model.eval()  # 关闭 Dropout

    x = torch.randn(batch_size, seq_len, config.hidden_size)

    # 位置编码
    freqs_cos, freqs_sin = precompute_rope(
        dim=config.hidden_size // config.num_attention_heads,
        end=x.shape[1],
        rope_base=config.rope_theta
    )
    position_inf = (freqs_cos, freqs_sin)

    # 前向传播
    output = model(x, position_inf)
    assert output.shape == (batch_size, seq_len, config.hidden_size), "输出形状错误"
    print("✅ 前向传播测试通过")

def test_kv_cache():
    """测试 KV 缓存机制"""
    config = TestConfig()
    model = GroupedQueryAttention(config)
    model.eval()

    # 初始输入
    x1 = torch.randn(1, 5, config.hidden_size)
    freqs_cos, freqs_sin = precompute_rope(
        dim=config.hidden_size // config.num_attention_heads,
        end=x1.shape[1],
        rope_base=config.rope_theta
    )
    position_inf = (freqs_cos, freqs_sin)

    # 第一次前向传播
    output1, kv_cache = model(x1, position_inf, use_cache=True)

    # 新增输入
    x2 = torch.randn(1, 3, config.hidden_size)
    freqs_cos, freqs_sin = precompute_rope(
        dim=config.hidden_size // config.num_attention_heads,
        end=x2.shape[1],
        rope_base=config.rope_theta
    )
    position_inf = (freqs_cos, freqs_sin)

    # 带缓存的前向传播
    output2, kv_cache = model(x2, position_inf, kv_cache=kv_cache, use_cache=True)

    # 验证缓存拼接
    assert kv_cache[0].shape[1] == 8, "KV 缓存长度错误"
    print("✅ KV 缓存测试通过")

def test_position_mask():
    """测试位置编码与掩码功能"""
    config = TestConfig()
    model = GroupedQueryAttention(config)
    model.eval()

    x = torch.randn(1, 8, config.hidden_size)
    freqs_cos, freqs_sin = precompute_rope(
        dim=config.hidden_size // config.num_attention_heads,
        end=x.shape[1],
        rope_base=config.rope_theta
    )
    position_inf = (freqs_cos, freqs_sin)

    # 创建掩码
    attention_mask = torch.tril(torch.ones(8, 8))  # 下三角掩码

    # 前向传播
    output = model(x, position_inf, attention_mask=attention_mask)

    # 验证掩码效果
    assert not torch.isnan(output).any(), "输出包含 NaN"
    print("✅ 位置编码与掩码测试通过")

def test_edge_cases():
    """测试边界条件"""
    config = TestConfig()
    model = GroupedQueryAttention(config)
    model.eval()

    # 测试不同序列长度
    for seq_len in [1, 5, 10, 20]:
        x = torch.randn(1, seq_len, config.hidden_size)
        freqs_cos, freqs_sin = precompute_rope(
            dim=config.hidden_size // config.num_attention_heads,
            end=x.shape[1],
            rope_base=config.rope_theta
        )
        position_inf = (freqs_cos, freqs_sin)
        output = model(x, position_inf)
        assert output.shape == (1, seq_len, config.hidden_size), f"序列长度 {seq_len} 测试失败"

    # 测试不同批大小
    for batch_size in [1, 2, 4]:
        x = torch.randn(batch_size, 5, config.hidden_size)
        freqs_cos, freqs_sin = precompute_rope(
            dim=config.hidden_size // config.num_attention_heads,
            end=x.shape[1],
            rope_base=config.rope_theta
        )
        position_inf = (freqs_cos, freqs_sin)
        output = model(x, position_inf)
        assert output.shape == (batch_size, 5, config.hidden_size), f"批大小 {batch_size} 测试失败"

    print("✅ 边界条件测试通过")

if __name__ == "__main__":
    print("开始测试 GroupedQueryAttention 模块...")
    test_forward()
    test_kv_cache()
    test_position_mask()
    test_edge_cases()
    print("🎉 所有测试通过")