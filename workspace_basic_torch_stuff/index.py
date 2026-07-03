import torch
import torch.nn as nn
from icecream import ic
ic.configureOutput(includeContext=True)
a = nn.Linear(2,3)
x = torch.randn(2)
x_prime = torch.randn(1,2)

ic(type(x))
#x.backward()

x69 = torch.randn(69,2)
x8769 = torch.randn(87,69,2)

def s1(i1,i2):
  ic(i1,type(i2))
def s2(i1,i2,i3):
  ic(i1,type(i2),i3.shape)
def b1(i1,i2):
  ic(i1,i2)
def b2(i1,i2,i3):
  ic(i1,i2,i3)

a.register_forward_hook(s2)
a.register_forward_pre_hook(s1)
a.register_full_backward_hook(b2)
a.register_full_backward_pre_hook(b1)


y = a(x)
ic(a._forward_hooks)

y.backward()

loss=y.sum()
ic(type(loss))
loss.backward()

