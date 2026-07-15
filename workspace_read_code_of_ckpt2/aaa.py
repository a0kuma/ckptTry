import torch

x = torch.tensor(2.0, requires_grad=True)
y = x * x

def show(where):
    print(where, torch._C._current_graph_task_id())

def backward_prehook(grad_outputs):
    show("inside backward")
    return grad_outputs

# Attach a hook to the MulBackward node.
y.grad_fn.register_prehook(backward_prehook)

show("before backward")

y.backward(retain_graph=True)

show("between backward calls")

x.grad = None
y.backward()

show("after backward")