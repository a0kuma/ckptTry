import os
os.environ["CUBLASLT_WORKSPACE_SIZE"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":0:0"
import torch
import torch.nn as nn
from icecream import ic
ic.configureOutput(includeContext=True)

torch.cuda.memory._record_memory_history()

z = nn.Sequential(
  nn.Linear(2,7,bias=False,device='cuda'),
  nn.Linear(7,3,bias=False,device='cuda')
  )
x = torch.randn(2,device='cuda')
y1 = z(x)

ic(dir(y1.grad_fn))
ic(y1.grad_fn._saved_dim)
ic(type(y1.grad_fn._saved_dim))
ic(y1.grad_fn._saved_self_sym_sizes)
ic(type(y1.grad_fn._saved_self_sym_sizes))

torch.cuda.memory._dump_snapshot('abc.pickle')
