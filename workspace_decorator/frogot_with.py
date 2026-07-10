class a():
  def __init__(self,x):
    print(x)
  def __enter__(self):
    return 123
  def __exit__(self,a,b,c):
    print('x')
  def __call__(self,t):
    print(t)
    return self
c=a(789)
with a(456) as b:
  print(b)
