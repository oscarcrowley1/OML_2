import math
from matplotlib import projections
import sympy
import matplotlib.pyplot as plt
import numpy as np
from torch import linspace
from mpl_toolkits.mplot3d import Axes3D
#x0, x1 = sympy.symbols("x0, x1", real=True)

from sympy.plotting import plot3d


#gamma = 1


x0, y0 = sympy.symbols("x0, y0", real=True)
#x0=sympy.Array([x00,x01])
#gamma_func=g*(x00**2)
#func= 2*x0*x0 + y0*y0
func= 8*(x0-10)**4+9*(y0-0)**2
#func= sympy.Max(x0-10,0)+9*sympy.Abs(y0-0)
x_deriv = sympy.diff(func, x0)
y_deriv = sympy.diff(func, y0)
print(func,x_deriv,y_deriv)

epsilon = 0.0001
fstar = 0

def f(xy):
    return func.subs([(x0, xy[0]), (y0, xy[1])])

def dfdx(xy):
    return x_deriv.subs([(x0, xy[0]), (y0, xy[1])])

def dfdy(xy):
    return y_deriv.subs([(x0, xy[0]), (y0, xy[1])])

def calc_polyak(fxy, fstar, slope):
    return (fxy-fstar)/(slope.dot(np.transpose(slope)) + epsilon)

plot3d(func, (x0, -20, 20), (y0, -20, 20))

