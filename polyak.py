from matplotlib import projections
import sympy
import matplotlib.pyplot as plt
import numpy as np
from torch import linspace
from mpl_toolkits.mplot3d import Axes3D

x0, y0 = sympy.symbols("x0, y0", real=True)
func= 8*(x0-10)**4+9*(y0-0)**2
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

x_start_range = [0.01, 0.1, 1, 10, 100]

x_start = 1
y_start = 1
num_iterations = 100
alpha = 0.1

curr_xy = [x_start, y_start]

curr_z = f(curr_xy)

xy_guesses = []
z_values = []

for iteration in range(num_iterations):
    xy_guesses.append(curr_xy)
    z_values.append(curr_z)
    
        
    slope = np.array([dfdx(curr_xy), dfdy(curr_xy)])   
    alpha = calc_polyak(f(curr_xy), fstar, slope)
    step = slope*alpha
    
    curr_xy = curr_xy - step
    curr_z = f(curr_xy)

xy_guesses = np.array(xy_guesses)
z_values = np.array(z_values)
      
plt.scatter(xy_guesses[:, 0], xy_guesses[:, 1], c=range(num_iterations))
plt.show()