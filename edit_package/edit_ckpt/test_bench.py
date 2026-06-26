import os
os.environ["CUBLASLT_WORKSPACE_SIZE"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":0:0"

import torch
import torch.nn as nn
# Import from LOCAL modified copy instead of system torch
from checkpoint import checkpoint

torch.cuda.memory._record_memory_history()
model = nn.Sequential(
    nn.Linear(100, 2048, device='cuda'),
    nn.Linear(2048, 4096, device='cuda'),
    nn.Linear(4096, 256, device='cuda'),
    nn.Linear(256, 10, device='cuda')
)

# Dummy data
batch_size = 32
x = torch.randn(batch_size, 100, device='cuda')

y_prime = checkpoint(
    model,
    input=x,
    use_reentrant=False,
    eager_delete=True,           # <-- the new flag
)

loss_but_in_vec = torch.ones_like(y_prime)
y_prime.backward(loss_but_in_vec)

torch.cuda.memory._dump_snapshot('abc.pickle')

# --- Correctness check: compare grads with vanilla (no checkpoint) ---
print("\n=== Correctness Check ===")
model2 = nn.Sequential(
    nn.Linear(100, 2048, device='cuda'),
    nn.Linear(2048, 4096, device='cuda'),
    nn.Linear(4096, 256, device='cuda'),
    nn.Linear(256, 10, device='cuda')
)
# Copy weights
with torch.no_grad():
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        p2.copy_(p1)

x2 = x.detach().clone().requires_grad_(True)
x_check = x.detach().clone().requires_grad_(True)

# Vanilla forward+backward (no checkpoint at all)
y_vanilla = model2(x2)
y_vanilla.backward(torch.ones_like(y_vanilla))

# eager_delete checkpoint forward+backward
model3 = nn.Sequential(
    nn.Linear(100, 2048, device='cuda'),
    nn.Linear(2048, 4096, device='cuda'),
    nn.Linear(4096, 256, device='cuda'),
    nn.Linear(256, 10, device='cuda')
)
with torch.no_grad():
    for p1, p3 in zip(model.parameters(), model3.parameters()):
        p3.copy_(p1)

y_ckpt = checkpoint(
    model3,
    input=x_check,
    use_reentrant=False,
    eager_delete=True,
)
y_ckpt.backward(torch.ones_like(y_ckpt))

# Compare parameter gradients
all_close = True
for i, (p2, p3) in enumerate(zip(model2.parameters(), model3.parameters())):
    if not torch.allclose(p2.grad, p3.grad, atol=1e-5):
        print(f"  MISMATCH at param {i}: max diff = {(p2.grad - p3.grad).abs().max().item():.2e}")
        all_close = False
    else:
        print(f"  param {i}: OK (max diff = {(p2.grad - p3.grad).abs().max().item():.2e})")

if all_close:
    print("\n✓ All parameter gradients match!")
else:
    print("\n✗ Some gradients differ!")
