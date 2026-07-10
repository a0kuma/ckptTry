from icecream import ic
def d(p):
  #def g(z):
    #return p(z)
  def i(*args,**kargs):
    ic(args,kargs)
    return 30
  return i
@d
def m(z):
  return z*3
ic(m(8))
