from icecream import ic
class a():
  def __init__(self):
    self.x=3
  def b():
    return 4
  def c(self):
    return self.x
p=a()
ic(p.c())
#ic(p.b())
i=p.b.__func__
ic(i())
