from icecream import ic
#def a(b):
#  def c(d):
#    def f(g):
#      t = d(g*2)
#      return t+b 
#    return f
#  return c
class a():
  def __init__(self,b):
    self.the_call=b
  def __call__(self,d):
    t=self.the_call(d*2)
    return t+7
@a
def y(x):
  z = x*3
  return z
ic(y(5))

