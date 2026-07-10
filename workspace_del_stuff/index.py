from icecream import ic
#import torch
class a:
  def __init__(self):
    ic('2')
  #def __del__(self):
    #ic('3')
c=a()
ic(c)
d=c
del c
ic(d)
