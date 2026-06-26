import torch
x=torch.tensor([
    [1.0,2.0],[3.0,4.0]
    ],device='cuda')
print(x)
o=torch.ones_like(x)
print(o)
