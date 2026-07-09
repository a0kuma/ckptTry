from icecream import ic
ic.configureOutput(includeContext=True)
class tina1(object):
  def __init__(self,a):
    self.a=a
    ic('start the init of class')
    ic('end the init of class')
  def __enter__(self):
    ic('inside with, start of enter')
    ic('inside with, end of enter')
  def __exit__(self, type, value, traceback):
    ic('inside with, start of exit',self.a)
    #ic(type,value,traceback)
    ic('inside with, end of exit')

ic('create an object')
var_obj = tina1(87)
ic(var_obj)
ic('start of with')
with var_obj:#tina1():
  ic('inside with')
  ic(var_obj)
  #raise ValueError('A very specific bad thing happened.')
  #ic("after error")
ic('out of with')
