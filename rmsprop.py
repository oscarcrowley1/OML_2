import math
from turtle import color
from matplotlib import projections
import sympy
import matplotlib.pyplot as plt
import numpy as np
from torch import linspace
from mpl_toolkits.mplot3d import art3d

x0, y0 = sympy.symbols("x0, y0", real=True)
func_num = 1

if func_num == 1:
    func= 8*(x0-10)**4+9*(y0-0)**2
    alpha_range = [0.01]#[0.00001, 0.0001, 0.001, 0.01]#, 0.1, 1]
    x_start = 1
    y_start = 1
else:
    func= sympy.Max(x0-10,0)+9*sympy.Abs(y0-0)
    alpha_range = [1]
    x_start = 15
    y_start = 10
    def cont_func(x, y):
        return np.maximum(x-10, 0)+9*(np.abs(y))

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

num_iterations = 1000

alpha_0 = 0.01
beta_range = [0.9, 0.25]

for alpha_0 in alpha_range:
    t = 1
    beta = 0.9
    sum = 0

    curr_alpha = alpha_0

    curr_xy = [x_start, y_start]

    curr_z = f(curr_xy)

    xy_guesses = []
    z_values = []

    for iteration in range(num_iterations):
        xy_guesses.append(curr_xy)
        z_values.append(curr_z)
        slope = np.array([dfdx(curr_xy), dfdy(curr_xy)])
        curr_xy = curr_xy - curr_alpha*slope
        curr_z = f(curr_xy)
        
        sum = beta*sum + (1-beta)*(slope.dot(np.transpose(slope)))
        curr_alpha = alpha_0/(math.sqrt(sum) + epsilon)
        t = t+1

    xy_guesses = np.array(xy_guesses)
    z_values = np.array(z_values)
    plt.plot(z_values, label=f"beta={beta}")
    
plt.xlabel("# Iterations")
plt.ylabel("f(x, y)")
plt.title(f"RMSProp with beta={beta} and varying alpha")
plt.legend()
plt.yscale('log')
plt.show()