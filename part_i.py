import sympy
import matplotlib.pyplot as plt
import numpy as np
from torch import linspace
#x0, x1 = sympy.symbols("x0, x1", real=True)


#gamma = 1


x0, y0 = sympy.symbols("x0, y0", real=True)
#x0=sympy.Array([x00,x01])
#gamma_func=g*(x00**2)
func=x0*y0
x_deriv = sympy.diff(func, x0)
y_deriv = sympy.diff(func, y0)
print(func,x_deriv,y_deriv)

epsilon = 0.0001
fstar = 0

def f(x, y):
    return func.subs([(x0, x), (y0, y)])

def dfdx(x, y):
    return x_deriv.subs([(x0, x), (y0, y)])

def dfdy(x, y):
    return y_deriv.subs([(x0, x), (y0, y)])

def calc_polyak(fxy, fstar, slope):
    return (fx-fstar)/(slope.dot(np.transpose(slope)) + epsilon)

gamma_range = [0.001, 0.01, 0.1, 1, 10, 100]
#gamma_range = linspace(-10, 10, 10)
x_start_range = [0.01, 0.1, 1, 10, 100]

#for alpha in alpha_range:
gamma = 1
x_start = 1
num_iterations = 1000

alpha = 0.1

current_x = x_start
#current_x = 1
current_y = f(current_x, gamma)
print(current_y)

x_guesses = []
y_values = []

for iteration in range(num_iterations):
    
    if current_y > 10000000:
        break
    
    x_guesses.append(current_x)
    y_values.append(current_y)
    
    print(current_x)
    print(current_y)
    print("\n")
    
    prev_x = current_x
        
    slope = np.array([dfdx(current_x, current_y), dfdy(current_x, current_y)])
    step = calc_polyak(f(current_x, current_y), fstar, slope)
    current_x = current_x - step
    current_y = f(current_x, gamma)
            
        # plt.title("x and f(x)")
        # plt.xlabel("x")
        # plt.ylabel("f(x)")
        # plt.plot(x_guesses, y_values)
        # #plt.show()