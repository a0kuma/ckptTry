import os
import torch
import torch.nn as nn
os.environ["CUBLASLT_WORKSPACE_SIZE"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

torch.cuda.memory._record_memory_history()
model = nn.Sequential(
    nn.Linear(100, 4096, device='cuda'),
    nn.Linear(4096, 256, device='cuda'),
    nn.Linear(256, 10, device='cuda')
).to('cuda')
# Dummy data
batch_size = 32
x = torch.randn(batch_size, 100, device='cuda')

y_prime = model(x)

#loss=y_prime.sum()
loss_but_in_vec = torch.ones_like(y_prime)
y_prime.backward(loss_but_in_vec)
#loss.backward()
torch.cuda.memory._dump_snapshot('abc_no_ckpt.pickle')

