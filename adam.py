import math
from matplotlib import projections
import sympy
import matplotlib.pyplot as plt
import numpy as np
from torch import linspace
from mpl_toolkits.mplot3d import art3d

x0, y0 = sympy.symbols("x0, y0", real=True)
func_num=1
if func_num == 1:
    func= 8*(x0-10)**4+9*(y0-0)**2
    alpha_range = [1]
    x_start = 1
    y_start = 1
    def cont_func(x, y):
        return 8*np.power(x-10, 4)+9*np.power(y-0, 2)
else:
    func= sympy.Max(x0-10,0)+9*sympy.Abs(y0-0)
    alpha_range = [100, 10, 1, 0.1, 0.01, 0.001]
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
num_iterations = 2000

x_space = np.linspace(7, 11, 50)
y_space = np.linspace(-2.5, +2.5, 50)

X, Y = np.meshgrid(x_space, y_space)
Z = cont_func(X, Y)

for alpha in alpha_range:
    zeta_0 = 0
    t = 1
    beta1 = 0.9
    beta2 = 0.99
    sum = 0

    curr_zeta = zeta_0

    curr_xy = [x_start, y_start]
    print(curr_xy)

    slope = np.array([dfdx(curr_xy), dfdy(curr_xy)])

    curr_m = 0
    curr_v = 0

    curr_z = f(curr_xy)

    xy_guesses = []
    z_values = []
    step_sizes = []

    for iteration in range(num_iterations):      
        xy_guesses.append(curr_xy)
        z_values.append(curr_z)
        slope = np.array([dfdx(curr_xy), dfdy(curr_xy)])
        
        curr_m = beta1*curr_m + (1-beta1)*(slope)
        curr_v = beta2*curr_v + (1-beta2)*(np.square(slope))
        m_hat = curr_m/(1-beta1**t)
        v_hat = curr_v/(1-beta2**t)
        
        curr_xy = curr_xy - alpha*(m_hat/((np.power(v_hat, 0.5))+epsilon))
        step_xy = alpha*(m_hat/((np.power(v_hat, 0.5))+epsilon))        
        curr_z = f(curr_xy)
        t = t+1

    xy_guesses = np.array(xy_guesses)
    z_values = np.array(z_values)
    
    cp = plt.contourf(X, Y, Z)#, colors='black')
    plt.colorbar(cp, label="f(x, y)")
    plt.plot(xy_guesses[:, 0], xy_guesses[:, 1], color='r', label="Descent")
    plt.xlabel("x Value")
    plt.ylabel("y Value")
    plt.legend()
    plt.show()
    
plt.plot(step_sizes, label=f"alpha={alpha}")

plt.xlabel("# Iterations")
plt.ylabel("f(x, y)")
plt.title(f"Adam with beta1={beta1}, beta2={beta2}, and varying alpha")
plt.legend()
plt.yscale('log')
plt.show()