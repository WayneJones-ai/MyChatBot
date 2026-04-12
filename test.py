import torch

# 1. 检查 CUDA 是否真的可用
print(f"CUDA 是否可用: {torch.cuda.is_available()}")

# 2. 查看你的显卡型号
if torch.cuda.is_available():
    print(f"当前显卡: {torch.cuda.get_device_name(0)}")
    print(f"CUDA 版本: {torch.version.cuda}")