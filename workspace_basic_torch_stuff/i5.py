import os
os.environ["CUBLASLT_WORKSPACE_SIZE"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":0:0"
import torch
import torch.nn as nn
from icecream import ic
ic.configureOutput(includeContext=True)

torch.cuda.memory._record_memory_history()

z = nn.Sequential(
  nn.Linear(2,5,bias=False,device='cuda'),
  nn.Linear(5,4,bias=False,device='cuda'),
  nn.Linear(4,3,bias=False,device='cuda')
  )

x = torch.randn(1,2,device='cuda')

def p(i):
  #return i
  if i.shape[0] == 1:
    return i.shape
  else:  
    return i
def u(i):
  if isinstance(i,torch.Size):
    return torch.ones(i,device='cuda')
  else:
    return i
  #return i

with torch.autograd.graph.saved_tensors_hooks(p,u):
  y = z(x)

#y = z(x)

l = torch.ones_like(y)

y.backward(l)

torch.cuda.memory._dump_snapshot('r2.pickle')
