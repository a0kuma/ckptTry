from icecream import ic
def a(b):
  #ic(b)
  def c(d):
    def f(g):
      t = d(g*2)
      return t+b 
    return f
  return c
@a(7)
def y(x):
  z = x*3
  return z
ic(y(5))

