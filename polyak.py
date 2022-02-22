from matplotlib import projections
import sympy
import matplotlib.pyplot as plt
import numpy as np
from torch import linspace
from mpl_toolkits.mplot3d import Axes3D
#x0, x1 = sympy.symbols("x0, x1", real=True)


#gamma = 1


x0, y0 = sympy.symbols("x0, y0", real=True)
#x0=sympy.Array([x00,x01])
#gamma_func=g*(x00**2)
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

gamma_range = [0.001, 0.01, 0.1, 1, 10, 100]
#gamma_range = linspace(-10, 10, 10)
x_start_range = [0.01, 0.1, 1, 10, 100]

#for alpha in alpha_range:
gamma = 1
x_start = 1
y_start = 1
num_iterations = 100



alpha = 0.1

curr_xy = [x_start, y_start]
#curr_x = 1
#curr_y = y_start
print(curr_xy)

curr_z = f(curr_xy)

xy_guesses = []
#y_guesses = []
z_values = []

for iteration in range(num_iterations):
    
    if curr_xy[0] > 10000000:
        break
    
    xy_guesses.append(curr_xy)
    z_values.append(curr_z)
    
    print(f"Current XY:\t{curr_xy}")
    # print(f"Current Y:\t{curr_y}")
    print(f"Current Z:\t{curr_z}")
    
        
    slope = np.array([dfdx(curr_xy), dfdy(curr_xy)])
    print(f"Slope:\t{slope}")
    
    alpha = calc_polyak(f(curr_xy), fstar, slope)
    print(f"ALPHA:\t{alpha}")
    
    step = slope*alpha
    print(f"STEP:\t{step}")
    
    curr_xy = curr_xy - step
    curr_z = f(curr_xy)
    
    print("\n")
    # curr_x = curr_x - step
    # curr_y = f(curr_x, gamma)
            
        # plt.title("x and f(x)")
        # plt.xlabel("x")
        # plt.ylabel("f(x)")
        # plt.plot(x_guesses, y_values)
        # #plt.show()

xy_guesses = np.array(xy_guesses)
z_values = np.array(z_values)
      
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
plt.scatter(xy_guesses[:, 0], xy_guesses[:, 1], c=range(num_iterations))
# plt.scatter(xy_guesses[:, 0], xy_guesses[:, 1], c=z_values)
plt.show()