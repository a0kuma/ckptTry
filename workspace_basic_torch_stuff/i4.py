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

x = torch.randn(1,2,device='cuda')
y = z(x)

l = torch.ones_like(y)

#t = y.grad_fn

#def s(q):
#  for n in dir(q):
#    if n.startswith('_saved') or n.startswith('_raw_saved'):
#      ic(n,getattr(q,n))

#def k(t):
#  for j in t.next_functions:
#    if j[0] is not None:
#      ic(j[0])
#      s(j[0])
#      if len(j[0].next_functions) != 0:
#        k(j[0])

#s(t)
#k(t)

del y.grad_fn._saved_self
del y.grad_fn._raw_saved_self

#y.backward(l)

torch.cuda.memory._dump_snapshot('abc.pickle')
