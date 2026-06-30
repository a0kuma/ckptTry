from icecream import ic
a = [1, 2, 3]
iter_a = iter(a)
for i in iter_a:
  ic(i)
  if i == 2:
    ic('break')
    break
for i in iter_a:
  ic(i)
