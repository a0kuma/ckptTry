from icecream import ic
def a(*args,**kargs):
  ic(args)
  ic(kargs)
a(1,2,3,a=4,b=5,c=6)
