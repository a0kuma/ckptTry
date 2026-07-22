import os
os.environ["CUBLASLT_WORKSPACE_SIZE"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":0:0"

import torch
import torch.nn as nn
from icecream import ic
ic.configureOutput(includeContext=True)

torch.cuda.memory._record_memory_history()

model = nn.Sequential(
    nn.Linear(2, 3, bias = False, device='cuda'),
    nn.Linear(3, 5, bias = False, device='cuda')
)

x = torch.randn(1, 2,  device='cuda')

def pack_hook(i):
  #return i
  if any( i.untyped_storage().data_ptr() == tmp_p.untyped_storage().data_ptr() for tmp_l in model for tmp_p in tmp_l.parameters()):
    ic(i)
    return torch.ones(3,6,device='cuda')
  else:
    ic(i)
    return i

def unpack_hook(i):
  #return i
  if hasattr( i, 'untyped_storage' ) and any( i.untyped_storage().data_ptr() == tmp_p.untyped_storage().data_ptr() for tmp_l in model for tmp_p in tmp_l.parameters()):
    ic(i)
    return i
  else:
    ic(i)
    return i

with torch.autograd.graph.saved_tensors_hooks(
  pack_hook,
  unpack_hook,
):
  y = model(x) 
  loss = torch.ones_like(y)
y.backward(loss)

torch.cuda.memory._dump_snapshot('abc_wo_if.pickle')
