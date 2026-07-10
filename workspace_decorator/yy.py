from icecream import ic
def yyy():
  ic('y1')
  yield 
  ic('y2')

y4=yyy()
next(y4)
ic('?')
try:
  next(y4)
except StopIteration as e:
  ic('stop',e)
except Exception as e:
  ic(e)
finally:
  ic('XXX')

