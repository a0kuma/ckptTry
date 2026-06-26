import torch

# 1. 建立一個 Tensor，並大喊：『我要算它的梯度！』
x = torch.tensor([3.0], requires_grad=True)

# 2. 進行運算（這會偷偷在 C++ 底層蓋一塊反向磚塊）
y = x * 2
z = y + 5

print("================ 正向產物 ================")
print(f"x 的數值: {x}")
print(f"y 的數值: {y}")
print(f"z 的數值: {z}\n")

print("================ 揭開反向工頭的面紗 ================")
# grad_fn 就是指向 C++ 底層那個 Backward Node 的指標！
print(f"z 的反向工頭: {z.grad_fn}")
print(f"y 的反向工頭: {y.grad_fn}")
print(f"x 的反向工頭: {x.grad_fn} (因為 x 是葉子節點輸入，所以沒有上一關)")
