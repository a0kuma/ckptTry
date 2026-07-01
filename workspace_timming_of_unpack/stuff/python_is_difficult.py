# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 20:23:33 2026

@author: ai
"""
from icecream import ic

def a():
    ic('into a')
    yield 10
    ic('after y1')
    yield 20
    ic('after y2')
    yield 30
    ic('after y3')
    
def c():
    ic('into a')
    #yield 10
    ic('after y1')
    #yield 20
    ic('after y2')
    #yield 30
    ic('after y3')
    return 40

z=[1,2,3]
y=iter(z)

b=a()
ic('after init')
ic(next(b))
ic(next(b))
ic(next(b))
#ic(next(b))

ic(type(c))
ic(type(c()))
ic(type(a))
ic(type(b))

ic(type(z))
ic(type(y))

d_b=dir(b)
d_z=dir(z)
d_y=dir(y)
