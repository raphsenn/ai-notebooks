"""
Optimization problem:

    f: R² -> R with f(x, y) = x + y -> min(f)
    s.t. c(x, y) = 2 - x² - y² = 0

    Lagrangian is defined as L(x, y, a) = f(x, y) - a * c(x, y)

    Applied to f and c we get for our lagrangian function:
    L(x, y, a) = x + y - a * (2 - x² - y²)
 
    grad_L(x, y, a) = (1 + 2ax, 1 + 2ay, 2 - x² - y²)^t (t for transpose)
    
    (Now the Jacobian)
    grad²_L(x, y, a) = [[2a, 0, -2x],
                        [0, -2a, -2y],
                        [-2x, -2y, 0]]

Now translate this in easy python code and start newtons-method with x = 0, y = -2, a = 1.
"""
import numpy as np
from matplotlib import pyplot as plt
import time

def f(x, y):
    """
    f: R² -> R with f(x, y) = x + y
    """
    return x + y

def c(x, y):
    return 2 - x * x - y * y


def grad_L(x, y, a):
    return np.array([[1 + 2*x*a], [1 + 2*y*a], [2 - x * x - y * y]])


def Jacobian_grad_L(x, y, a):
    return np.array([[2 * a, 0 , 2 * x], [0, 2 * a, 2 * y], [-2 * x, - 2 * y, 0]])

def optimize(z_0, max_iterations: int, plot: bool):
    if plot:
        x = np.linspace(-3, 3, 400)
        y = np.linspace(-3, 3, 400)
        X, Y = np.meshgrid(x, y)
        F = f(X, Y)
        C = c(X, Y)
        plt.contour(X, Y, F, levels=20, cmap='viridis', label='Zielfunktion')
        plt.contour(X, Y, C, levels=[0], colors='red', label='Gleichheitsbeschränkung')
        current_pos = (z_0[0], z_0[1])

    for iteration in range(max_iterations): 
        F_k = grad_L(z_0[0], z_0[1], z_0[2])
        J_k = Jacobian_grad_L(z_0[0], z_0[1], z_0[2])
        solution = np.linalg.solve(J_k, -F_k)
        z_0 = np.matrix(z_0).T + solution
        z_0 = np.array(z_0).flatten()
        if plot: 
            plt.scatter(z_0[0], z_0[1], color='green', marker='x', label=f'iteration {iteration}')
            plt.text(z_0[0], z_0[1], str(iteration), color='black', fontsize=8, ha='right', va='bottom')
    if plot: 
        plt.scatter(z_0[0], z_0[1], color='green', marker='x', label='optimum') 
        plt.show() 
    return z_0


if __name__ == '__main__':
    z_0 = np.array([0, -2, 1]) # Initial guess.
    optimize(z_0, 10, True)
