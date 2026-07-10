# target: with WTF as WTF: where WTF no have enter and exit
# inside with need be a class w/ enter and exit
# the org input is a def(aka func)
# the wraper is a fun w/ io be func
from icecream import ic
class x():
  def __init__(self,p):
    self.p=p
  def __call__(self,f):
    if callable(f):
      self.f=f
      return self
    else:
      self.u=self.f(f)
      return self
  def __enter__(self):
    next(self.u)
    return 258
  def __exit__(self,a,b,c):
    try:
      next(self.u)
    except StopIteration as e:
      ic(e)
    finally:
      ic(self.p)
  def qq():
    return 369
@x(99)
def a(c):
  s=c+1
  ic('>a',s)
  s=s+1
  yield
  s=s+1
  ic('<a',s)
with a(3) as b:
  ic(b)

