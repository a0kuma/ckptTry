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
    nn.Linear(2, 7, bias = False, device='cuda'),
    nn.Linear(7, 3, bias = False, device='cuda'),
    nn.Linear(3, 5, bias = False, device='cuda')
)

x = torch.randn(1, 2,  device='cuda')

y = checkpoint.checkpoint(
    model,
    input=x,
    determinism_check='none',
    debug=False,
    early_stop=False,
    use_reentrant=False
)

loss = torch.ones_like(y)
y.backward(loss)

torch.cuda.memory._dump_snapshot('abc_new_song.pickle')
