from icecream import ic

def g():
    ic(10)
    yield(20)
    ic(30)
    yield(40)
gg=g()
ic(next(gg))
ic(next(gg))
#ic(next(gg))
ll=[1,2,3]
li=iter(ll)
ic(type(li))
ic(next(li))
ic(next(li))
ic(next(li))
#ic(next(li))
ic(type(gg))
