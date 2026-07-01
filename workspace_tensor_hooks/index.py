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
ic(y.grad_fn)
ic(type(y.grad_fn))
ic(y.grad_fn.next_functions)
ic(type(y.grad_fn.next_functions))
ic(y.grad_fn.next_functions[0])
ic(type(y.grad_fn.next_functions[0]))
ic(y.grad_fn.next_functions[0][0])
ic(type(y.grad_fn.next_functions[0][0]))
ic(dir(y.grad_fn.next_functions[0][0]))

ic(y.grad_fn.next_functions[0][0].next_functions)

ic(dir(y.grad_fn.next_functions[0][0]._input_metadata))
ic(y.grad_fn.next_functions[0][0]._input_metadata)
ic(y.grad_fn.next_functions[0][0].variable)
