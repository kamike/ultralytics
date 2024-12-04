import torch

# 假设 kp 是一个包含张量的列表或其他可迭代对象
kp = [torch.tensor([142.3645, 441.8608])]

# 获取 kp[0] 的第一个和第二个元素
first_value = kp[0][0].item()
second_value = kp[0][1].item()

print("第一个值:", first_value)
print("第二个值:", second_value)
