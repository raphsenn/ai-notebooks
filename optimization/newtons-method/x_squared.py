import numpy as np
from matplotlib import pyplot as plt


def f(x: np.ndarray) -> np.ndarray:
    """
    Represents the function f: R -> R with f(x) = x², where R is the set of real numbers.

    Parameters:
    x (np.ndarray): Input value.

    Returns:
    float: Result of the function f(x) = x².
    """
    return x * x 


def grad_f(x: np.ndarray) -> np.ndarray:
    """
    Represents the gradient function f': R -> R with f'(x) = 2*x.

    Parameters:
    x (np.ndarray): Input value.

    Returns:
    float: Result of the gradient function f'(x) = 2*x.
    """
    return 2 * x


def hess_f(x: np.ndarray) -> np.ndarray:
    """
    Represents the Hessian matrix for the function f: R -> R with Hessian(f) = 2,
    where f(x) = x² and R is the set of real numbers.

    Parameters:
    x (np.ndarray): Input value.

    Returns:
    float: Constant Hessian value, as the second derivative of f(x) is a constant, 
           Hessian(f) = 2 for any x in the real numbers.
    """ 
    return 2


def optimize(x0: float, tau: float, max_iterations: int, plot: bool):
    """
    Simple implementation of newtons-method optimization algorithm.

    f is a quadratic function, newtons-method converges in one iteration.

    Gradient descent: d_k = -grad_f(x_k)
    Newton-Method: d_k = -hess_f(f(x_k))^(-1) * grad_f(x_k)   
    """ 
    # Data for plotting. 
    x = np.linspace(-10, 10, 1000)
    y = f(x) 
    
    if plot:
        # Plot setup.
        plt.figure(num="f(x)=x²")
        plt.xlabel("x")
        plt.ylabel("y")

    # Start newtons-method
    current_pos = (x0, f(x0))
    iterations = 0
    for _ in range(max_iterations):
        iterations += 1 
        
        # Update x0 using: x_k + tau * (-hess_f(f(x_k))^(-1) * grad_f(x_k))
        x0 = x0 + tau * (-hess_f(x0) ** (-1)) * grad_f(x0)
        
        # Update current position. 
        current_pos = (x0, f(x0)) 
        
        if plot:
            # Plotting.
            plt.plot(x, y)
            plt.scatter(current_pos[0], current_pos[1], color="red")
            plt.pause(0.0001)
            plt.clf() 
        
        if x0  == 0:
            break
 
    return x0, iterations

if __name__ == '__main__':
    opt = optimize(10, 1, 100, True)
    print(f"optimum = {opt[0]}, iterations = {opt[1]}")
