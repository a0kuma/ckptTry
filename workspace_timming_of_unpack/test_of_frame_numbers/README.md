# CODE

https://github.com/a0kuma/ckptTry/commit/cd6f1b3eaad542e705be90d03b285aeabdf31225

# RESULT

```
(sum26) andy@140:~/playground/ckptTry/workspace_timming_of_unpack$ python tb_basic.py
ic| checkpoint.py:1700 in _checkpoint_without_reentrant_generator()
    new_frame: <checkpoint._CheckpointFrame object at 0x7d0a244396a0>
/home/andy/anaconda3/envs/sum26/lib/python3.14/site-packages/torch/nn/modules/linear.py:134: UserWarning: Attempting to run cuBLAS, but there was no current CUDA context! Attempting to set the primary context... (Triggered internally at /pytorch/aten/src/ATen/cuda/CublasHandlePool.cpp:370.)
  return F.linear(input, self.weight, self.bias)
(sum26) andy@140:~/playground/ckptTry/workspace_timming_of_unpack$ python tb_basic.py
ic| checkpoint.py:1700 in _checkpoint_without_reentrant_generator()
    new_frame: <checkpoint._CheckpointFrame object at 0x7adfb5ad56a0>
ic| checkpoint.py:1700 in _checkpoint_without_reentrant_generator()
    new_frame: <checkpoint._CheckpointFrame object at 0x7adfb51ae210>
/home/andy/anaconda3/envs/sum26/lib/python3.14/site-packages/torch/nn/modules/linear.py:134: UserWarning: Attempting to run cuBLAS, but there was no current CUDA context! Attempting to set the primary context... (Triggered internally at /pytorch/aten/src/ATen/cuda/CublasHandlePool.cpp:370.)
  return F.linear(input, self.weight, self.bias)
(sum26) andy@140:~/playground/ckptTry/workspace_timming_of_unpack$
```

# TO SUM UP

if create 1 segmwnt will have 1 frame, 2 segment will have 2 frame, and about how to create segment is like this 


```python=
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
    #nn.Linear(100, 4096, device='cuda', bias=False),
    #nn.Linear(1024, 4096, device='cuda', bias=False),
    nn.Linear(4096, 256, device='cuda', bias=False),
    nn.Linear(256, 10, device='cuda', bias=False)
)

# Dummy data
batch_size = 32
x = torch.randn(batch_size, 100, requires_grad = False , device='cuda')

y_prime1 = checkpoint.checkpoint(
    nn.Linear(100, 4096, device='cuda', bias=False),
    input = x,
    early_stop = False,
    determinism_check = "none",
    use_reentrant = False
)
y_prime2 = checkpoint.checkpoint(
    model,
    input = y_prime1,
    early_stop = False,
    determinism_check = "none",
    use_reentrant = False
)



loss_but_in_vec = torch.ones_like(y_prime2)
y_prime2.backward(loss_but_in_vec)

torch.cuda.memory._dump_snapshot('abc.pickle')

```
