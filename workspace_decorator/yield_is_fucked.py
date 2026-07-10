from icecream import ic
def a(x):
  ic(x)
  t = yield x+3
  ic(t)
  yield t+x
b=a(6)
ic('---')
#ic(next(b))
ic(b.send(None))
ic(b.send(7))
