import os
os.environ["CUBLASLT_WORKSPACE_SIZE"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":0:0"

import torch
import torch.nn as nn
#import torch.utils.checkpoint as checkpoint
import checkpoint

from icecream import ic 
ic.configureOutput(includeContext=True)

torch.cuda.memory._record_memory_history()
model = nn.Sequential(
    nn.Linear(100, 4096, device='cuda', bias=False),
    #nn.Linear(1024, 4096, device='cuda', bias=False),
    nn.Linear(4096, 256, device='cuda', bias=False),
    nn.Linear(256, 10, device='cuda', bias=False)
)

# Dummy data
batch_size = 32
x = torch.randn(batch_size, 100, requires_grad = False , device='cuda')

y_prime = checkpoint.checkpoint(
    model,#nn.Linear(100, 4096, device='cuda', bias=False),
    input = x,
    early_stop = False,
    determinism_check = "none",
    use_reentrant = False
)
#y_prime2 = checkpoint.checkpoint(
#    model,
#    input = y_prime1,
#    early_stop = False,
#    determinism_check = "none",
#    use_reentrant = False
#)



loss_but_in_vec = torch.ones_like(y_prime)
y_prime.backward(loss_but_in_vec)

torch.cuda.memory._dump_snapshot('abc.pickle')
