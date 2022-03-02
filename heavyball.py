import math
from matplotlib import markers, projections
import sympy
import matplotlib.pyplot as plt
import numpy as np
from torch import linspace
from mpl_toolkits.mplot3d import art3d

x0, y0 = sympy.symbols("x0, y0", real=True)

func_num=2
if func_num == 1:
    func= 8*(x0-10)**4+9*(y0-0)**2
    alpha_range = [0.000001, 0.00001, 0.0001, 0.001, 0.01]#, 0.1, 1]
    #alpha_range = np.logspace(-10, -3, 8)
    x_start = 1
    y_start = 1
    def cont_func(x, y):
        return 8*np.power(x-10, 4)+9*np.power(y-0, 2)
    
else:
    func= sympy.Max(x0-10,0)+9*sympy.Abs(y0-0)
    alpha_range = [10, 1, 0.1, 0.01, 0.001]
    alpha_range = [0.001]
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

x_start_range = [0.01, 0.1, 1, 10, 100]
num_iterations = 1000

x_space = np.linspace(5, 20, 50)
y_space = np.linspace(-5, +5, 50)

X, Y = np.meshgrid(x_space, y_space)
Z = cont_func(X, Y)

for alpha in alpha_range:


    zeta_0 = 0
    t = 0
    beta = 0.9
    sum = 0

    #alpha = 0.0001

    curr_zeta = zeta_0

    curr_xy = [x_start, y_start]
    #curr_x = 1
    #curr_y = y_start
    print(curr_xy)

    curr_z = f(curr_xy)

    xy_guesses = []
    #y_guesses = []
    z_values = []

    for iteration in range(num_iterations):
        print(f"Iteration:\t{iteration}")
        
        xy_guesses.append(curr_xy)
        z_values.append(curr_z)
        
        slope = np.array([dfdx(curr_xy), dfdy(curr_xy)])
        curr_zeta = beta*curr_zeta + alpha*slope
        
        curr_xy = curr_xy - curr_zeta
        curr_z = f(curr_xy)
        t = t+1

    xy_guesses = np.array(xy_guesses)
    z_values = np.array(z_values)

    cp = plt.contourf(X, Y, Z)#, colors='black')
    plt.colorbar(cp, label="f(x, y)")
    plt.plot(xy_guesses[:, 0], xy_guesses[:, 1], color='r', label="Descent")
    
    plt.scatter(xy_guesses[170, 0], xy_guesses[170, 1], label="First Bump", c="m", markers="x")
    plt.scatter(xy_guesses[550, 0], xy_guesses[550, 1], label="Settling", c="y", markers="x")
    plt.xlabel("x Value")
    plt.ylabel("y Value")
    plt.legend()
    plt.show()
    
plt.xlabel("# Iterations")
plt.ylabel("f(x, y)")
plt.title(f"HeavyBall with beta={beta} and varying alpha")
plt.legend()
plt.yscale('log')
plt.show()