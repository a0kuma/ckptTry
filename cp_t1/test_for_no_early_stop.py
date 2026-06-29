import os
os.environ["CUBLASLT_WORKSPACE_SIZE"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":0:0"

import torch
import torch.nn as nn
#import torch.utils.checkpoint as checkpoint
import checkpoint

torch.cuda.memory._record_memory_history()
model = nn.Sequential(
    nn.Linear(100, 4096, device='cuda'),
    #nn.Linear(2048, 4096, device='cuda'),
    nn.Linear(4096, 256, device='cuda'),
    nn.Linear(256, 10, device='cuda')
)

# Dummy data
batch_size = 32
x = torch.randn(batch_size, 100, device='cuda')

y_prime = checkpoint.checkpoint(
    model,
    input = x,
    early_stop = False,
    use_reentrant = False
)

loss_but_in_vec = torch.ones_like(y_prime)
y_prime.backward(loss_but_in_vec)

torch.cuda.memory._dump_snapshot('early_stop_False4.pickle')
