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

for name, p in model.named_parameters():
  print(
    name,
    "shape =", tuple(p.shape),
    "device =", p.device,
    "addr =", hex(p.data_ptr()),
  )

def pack_hook(i):
  ic(hex(i.data_ptr()))
  #ic([t for t in dir(i) if not t.startswith('__')])
  return i

def unpack_hook(i):
  #ic(i)
  return i

with torch.autograd.graph.saved_tensors_hooks(
  pack_hook,
  unpack_hook,
):
  y = model(x) 
  loss = torch.ones_like(y)
y.backward(loss)

torch.cuda.memory._dump_snapshot('abc.pickle')
