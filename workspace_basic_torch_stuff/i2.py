import os
os.environ["CUBLASLT_WORKSPACE_SIZE"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":0:0"
import torch
import torch.nn as nn
from icecream import ic
ic.configureOutput(includeContext=True)

torch.cuda.memory._record_memory_history()

a = nn.Linear(2,3,bias=False,device='cuda')
x = torch.randn(2,device='cuda')
c = None
d = None
e = None

def s1(i1,i2):
  global c
  global e
  ic('->->->->|||||--------')
  c=torch.randn(50,device='cuda')
  e=c
#  ic(i1,type(i2))
def s2(i1,i2,i3):
  global c
  ic('--------|||||->->->->')
  del c
#  ic(i1,type(i2),i3.shape)
def b1(i1,i2):
  global d
  ic('--------|||||<-<-<-<-')
  d=torch.randn(60,device='cuda')
#  ic(i1,i2)
def b2(i1,i2,i3):
  global e  
  global d
  ic('<-<-<-<-|||||--------')
  del e
  del d
#  ic(i1,i2,i3)

def gfh_p1(i1):
  ic('grad fn hook pre')
def gfh1(i1,i2):
  ic('grad fn hook')
def gfh_p2(i1):
  ic('grad fn hook pre')
def gfh2(i1,i2):
  ic('grad fn hook')
def gfh_p3(i1):
  ic('grad fn hook pre')
def gfh3(i1,i2):
  ic('grad fn hook')
def gfh_p4(i1):
  ic('grad fn hook pre')
def gfh4(i1,i2):
  ic('grad fn hook')
def gfh_p5(i1):
  ic('grad fn hook pre')
def gfh5(i1,i2):
  ic('grad fn hook')

#a.register_forward_hook(s2)
#a.register_forward_pre_hook(s1)
#a.register_full_backward_hook(b2)
#a.register_full_backward_pre_hook(b1)

y = a(x)
loss = torch.ones_like(y)

ic(y.grad_fn)
#ic(y.grad_fn.next_functions[0][0])
#ic(y.grad_fn.next_functions[0][0].next_functions[0][0])
#ic(y.grad_fn.next_functions[0][0].next_functions[0][0].next_functions[1][0])
#ic(y.grad_fn.next_functions[0][0].next_functions[0][0].next_functions[1][0].next_functions[0][0])


y.grad_fn.register_prehook(gfh_p1)
y.grad_fn.next_functions[0][0].register_prehook(gfh_p2)
y.grad_fn.next_functions[0][0].next_functions[0][0].register_prehook(gfh_p3)
y.grad_fn.next_functions[0][0].next_functions[0][0].next_functions[1][0].register_prehook(gfh_p4)
y.grad_fn.next_functions[0][0].next_functions[0][0].next_functions[1][0].next_functions[0][0].register_prehook(gfh_p5)

y.grad_fn.register_hook(gfh1)
y.grad_fn.next_functions[0][0].register_hook(gfh2)
y.grad_fn.next_functions[0][0].next_functions[0][0].register_hook(gfh3)
y.grad_fn.next_functions[0][0].next_functions[0][0].next_functions[1][0].register_hook(gfh4)
y.grad_fn.next_functions[0][0].next_functions[0][0].next_functions[1][0].next_functions[0][0].register_hook(gfh5)

y.backward(loss)

torch.cuda.memory._dump_snapshot('abc.pickle')
