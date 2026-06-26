import os
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
os.environ["CUBLASLT_WORKSPACE_SIZE"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

torch.cuda.memory._record_memory_history()
model = nn.Sequential(
    nn.Linear(100, 4096, device='cuda'),
    nn.Linear(4096, 128, device='cuda'),
    nn.Linear(128, 10, device='cuda')
).to('cuda')
# Dummy data
batch_size = 32
x = torch.randn(batch_size, 100, device='cuda')

y_prime = checkpoint.checkpoint(#_sequential(
    model,
    #segments=1,
    input=x,
    use_reentrant=False
)
#crossEntropyLoss_func_obj=nn.CrossEntropyLoss()
#y = torch.randint(0, 10, (batch_size,), device='cuda')

loss=y_prime.sum()#crossEntropyLoss_func_obj(y_prime,y)
#dummy_input = torch.randn(4800, 100000, device='cuda')
loss.backward()
#del dummy_input
torch.cuda.memory._dump_snapshot('abc.pickle')
