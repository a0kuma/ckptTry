import os
os.environ["CUBLAST_WORKSPACE_SIZE"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":0:0"
import torch
import torch.nn as nn
from icecream import ic
ic.configureOutput(includeContext = True)
torch.cuda.memory._record_memory_history()
model = nn.Sequential(
  nn.Linear(2, 7, device = "cuda"),
  nn.Linear(7, 5, device = "cuda"),
)
x = torch.randn(3, 2, device = "cuda")
#y = model(x)
cc = torch.compile(model)
y = cc(x)
torch.cuda.memory._dump_snapshot("abc2.pickle")
