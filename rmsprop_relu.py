import math
from matplotlib import projections
import sympy
import matplotlib.pyplot as plt
import numpy as np
from torch import linspace
from mpl_toolkits.mplot3d import art3d
#x0, x1 = sympy.symbols("x0, x1", real=True)


#gamma = 1


x0, y0 = sympy.symbols("x0, y0", real=True)
#x0=sympy.Array([x00,x01])
#gamma_func=g*(x00**2)
#func= 2*x0*x0 + y0*y0
#func= 8*(x0-10)**4+9*(y0-0)**2
#func= sympy.Max(x0-10,0)+9*sympy.Abs(y0-0)
func_num = 0

if func_num == 1:
    func= 8*(x0-10)**4+9*(y0-0)**2
    alpha_range = [0.00001, 0.0001, 0.001, 0.01]#, 0.1, 1]
    #alpha_range = np.logspace(-10, -3, 8)
    x_start = 1
    y_start = 1
elif func_num == 2:
    func= sympy.Max(x0-10,0)+9*sympy.Abs(y0-0)
    alpha_range = [0.001, 0.01, 0.1, 1, 10, 100]
    x_start = 15
    y_start = 10
else:
    func = sympy.Max(x0, 0)
    alpha_range = [0.1]
    x_start = +100
    y_start = 0

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
num_iterations = 1100


for alpha_0 in alpha_range:
    # alpha_0 = 0.0001
    t = 0
    beta = 0.9
    sum = 0

    curr_alpha = alpha_0

    curr_xy = [x_start, y_start]
    #curr_x = 1
    #curr_y = y_start
    print(curr_xy)

    curr_z = f(curr_xy)

    xy_guesses = []
    #y_guesses = []
    z_values = []
    step_sizes = []

    for iteration in range(num_iterations):
        print(f"Iteration:\t{iteration}")
        
        #if abs(curr_xy[0]) > 10000000 or abs(curr_xy[1]) > 10000000:
        #    break
        
        xy_guesses.append(curr_xy)
        z_values.append(curr_z)
        
        print(f"Current XY:\t{curr_xy}")
        # print(f"Current Y:\t{curr_y}")
        print(f"Current Z:\t{curr_z}")
        
        slope = np.array([dfdx(curr_xy), dfdy(curr_xy)])
        print(f"Slope:\t{slope}")
        
        curr_xy = curr_xy - curr_alpha*slope
        step_sizes.append((curr_alpha*slope[0]))
        
        curr_z = f(curr_xy)
        print(f"New XY:\t{curr_xy}")
        # print(f"Current Y:\t{curr_y}")
        print(f"New Z:\t{curr_z}")
            
        sum = beta*sum + (1-beta)*(slope.dot(np.transpose(slope)))
        print(f"SUM:\t{sum}")

        
        curr_alpha = alpha_0/(math.sqrt(sum) + epsilon)

        print(f"ALPHA:\t{curr_alpha}")
        
        t = t+1
        
        #step = slope*alpha
        #print(f"STEP:\t{step}")
        
        
        
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
    # plt.scatter(xy_guesses[:, 0], xy_guesses[:, 1], c=range(iteration+1))
    # # plt.scatter(xy_guesses[:, 0], xy_guesses[:, 1], c=z_values)
    # plt.show()

    plt.plot(z_values, label=f"alpha={alpha_0}")
    #plt.plot(step_sizes, label=f"step={alpha_0}")

plt.xlabel("# Iterations")
plt.ylabel("f(x, y)")
plt.title(f"RMSProp with beta={beta} and alpha={alpha_0}")
plt.legend()
#plt.yscale('log')
plt.show()

# art3d.Line3D(xy_guesses[:, 0], xy_guesses[:, 1], z_values)
# plt.show()