from icecream import ic
import copy
import weakref
class a():
  def __init__(self,x):
    self.x=x

b=weakref.WeakKeyDictionary()
#ic(b)
a1=a(23)
b[a1]=a(45)
print(b.keys())
h={1:b}
c=copy.deepcopy(h)
print(h[1].keys())
print(c[1].keys())

