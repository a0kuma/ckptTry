from icecream import ic
import torch

ic.configureOutput(includeContext=True)

a = torch.randn(5,requires_grad=True)
b = torch.ones(5,requires_grad=True)

ic(a)
ic(b)

y = a * b

ic(y)
ic(dir(y.grad_fn))
ic(y.grad_fn.register_hook)
ic(dir(y.grad_fn.register_hook))
ic(type(y.grad_fn.register_hook))
ic(y.grad_fn.register_prehook)
ic(dir(y.grad_fn.register_prehook))
ic(type(y.grad_fn.register_prehook))
