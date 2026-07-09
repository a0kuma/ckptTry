from icecream import ic
class a():
  def __init__(self,a):#input ok output kindof
    self.a=a
    ic(self.a)
  def __enter__(self,a):#input ok output ok
    self.a=a
    ic(self.a)
    return self
  def __exit__(self,a,b,c):
    pass
  #def __call__(self,a):
    #return self


#b=a(3)
with a(2) as z:
  ic(z)
