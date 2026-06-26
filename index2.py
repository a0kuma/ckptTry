import torch

def print_autograd_graph(grad_fn, indent=""):
    if grad_fn is None:
        return
    
    # 印出當前工頭的名字
    print(f"{indent}└── 🧱 {grad_fn.__class__.__name__} (地址: {hex(id(grad_fn))})")
    
    # 遍歷這個工頭地圖上指向的下一關
    for next_fn, _ in grad_fn.next_functions:
        print_autograd_graph(next_fn, indent + "    ")

# 實驗程式碼
x = torch.tensor([3.0], requires_grad=True)
y = x * 2
z = y + 5

print("==== 遞迴打印 PyTorch C++ 底層的反向計算圖 ====")
print_autograd_graph(z.grad_fn)