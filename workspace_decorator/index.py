from icecream import ic
def a(b):
  ic(b)
  def y():
    ic(3)
  return y
@a
def c():
  ic(2)
ic(c())
