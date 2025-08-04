import torch

model = torch.jit.load("actor_only.pt")

# 打印模型结构
print(model)

# 尝试输入 [1, 270]
try:
    dummy = torch.zeros(1, 270)
    model(dummy)
    print("❌ 错误！模型仍然接收 270 维输入")
except Exception as e:
    print("✅ 正常拒绝 270 输入:", e)

# 尝试输入 [1, 45]
try:
    dummy = torch.zeros(1, 45)
    output = model(dummy)
    print("✅ 正常接受 45 输入，输出形状为:", output.shape)
except Exception as e:
    print("❌ 错误！模型不能接收 45 维:", e)