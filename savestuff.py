import numpy as np
import matplotlib.pyplot as plt

# for alpha in alpha_range:
def cont_func(x, y):
    return np.maximum(x-10, 0)+9*np.abs(y) #function 2
    #return 8*np.power(x-10, 4)+9*np.power(y-0, 2) #function 1
    
#function 1
x_space = np.linspace(-10, 30, 50)
y_space = np.linspace(-90, 90, 50)

#function 2
x_space = np.linspace(0, 40, 50)
y_space = np.linspace(-15, 15, 50)

X, Y = np.meshgrid(x_space, y_space)
Z = cont_func(X, Y)

cp = plt.contourf(X, Y, Z)#, colors='black')
plt.colorbar(cp, label="f(x, y)")
#plt.plot(xy_guesses[:, 0], xy_guesses[:, 1], color='r')
#plt.scatter(xy_guesses[:, 0], xy_guesses[:, 1], c=range(iteration+1))
# plt.scatter(xy_guesses[:, 0], xy_guesses[:, 1], c=z_values)
plt.xlabel("x Value")
plt.ylabel("y Value")
#plt.title("Movement of GD on the xy plane")
plt.show()

    # plt.plot(z_values, label=f"alpha={alpha_0}")

# plt.xlabel("# Iterations")
# plt.ylabel("f(x, y)")
# plt.title(f"RMSProp with beta={beta} and varying alpha")
# plt.legend()
# plt.yscale('log')
# plt.show()