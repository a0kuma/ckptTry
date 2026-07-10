from icecream import ic
class a():
  def __init__(self,b):
    self.b=b
  def __call__(self,d):
    if callable(d):
      self.f=d
      return self
    else:
      return self.f(d*self.b)+7
      
@a(2)
def y(x):
  z = x*3
  return z
ic(y(5))

