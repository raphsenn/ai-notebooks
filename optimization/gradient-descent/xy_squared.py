import numpy as np
from matplotlib import pyplot as plt


def f(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the value of the function f: R² -> R, where f(x, y) = x² + y².

    Parameters:
    - x (np.ndarray): The x-coordinate.
    - y (np.ndarray): The y-coordinate.

    Returns:
    float: The computed value of f(x, y).
    """
    return x**2 + y**2


def f_grad(x: np.ndarray, y: np.ndarray) -> tuple:
    """
    Compute the gradient of the function f': R² -> R, where f'(x, y) = (2x, 2y).

    Parameters:
    - x (np.ndarray): The x-coordinate.
    - y (np.ndarray): The y-coordinate.

    Returns:
    tuple: A tuple representing the gradient vector (2x, 2y).
    """
    return 2*x, 2*y



def norm(x: float, y: float) -> float:
    """
    Calculate the Euclidean norm (length) of a 2D vector.

    Parameters:
    - x (float): The x-component of the vector.
    - y (float): The y-component of the vector.

    Returns:
    float: The Euclidean norm of the vector (x, y).
    """
    return (x*x + y*y)**0.5


def optimize_without_backtracking(x0: float, y0: float, tau: float, max_iterations: int, plot: bool) -> tuple:
    """
    Perform gradient descent optimization without backtracking line search.

    Parameters:
    - x0 (float): Initial x-coordinate.
    - y0 (float): Initial y-coordinate.

    Returns:
    tuple: Optimized (x, y) coordinates.
    """
    # Set up the meshgrid for plotting the function surface
    x = np.linspace(-1, 1, 1000)    
    y = np.linspace(-1, 1, 1000)    
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
   
    if plot:
        # Create a 3D plot 
        plt.figure(num="f(x, y) = x² + y²")
        ax = plt.subplot(projection="3d", computed_zorder=False)
        ax.plot_surface(X, Y, Z, cmap="viridis")  
        
    # Start gradient descent
    current_pos = (x0, y0, f(x0, y0))
    for _ in range(max_iterations):
        X_grad, Y_grad = f_grad(x0, y0) 
        x0 = x0 - tau * X_grad
        y0 = y0 - tau * Y_grad
        current_pos = (x0, y0, f(x0, y0))
      
        if plot:
            # Plotting
            ax.plot_surface(X, Y, Z, cmap="viridis", zorder=0)  
            ax.scatter(current_pos[0], current_pos[1], current_pos[2], color="red", zorder=0)
            plt.pause(0.01)
            ax.clear()
        
    return x0, y0


def optimize_with_backtracking(x0: float, y0: float, eps: float, beta: float, max_iterations: int, plot: bool) -> tuple:
    """
    Perform gradient descent optimization with backtracking line search.

    Parameters:
    - x0 (float): Initial x-coordinate.
    - y0 (float): Initial y-coordinate.
    - eps (float):
    - beta (float):

    Returns:
    tuple: Optimized (x, y) coordinates.
    """ 
    # Set up the meshgrid for plotting the function surface
    x = np.linspace(-1, 1, 1000)    
    y = np.linspace(-1, 1, 1000)    
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
   
    if plot:
        # Create 3D plot 
        plt.figure(num="f(x, y) = x² + y²")
        ax = plt.subplot(projection="3d", computed_zorder=False)
        ax.plot_surface(X, Y, Z, cmap="viridis")  
        
    # Start gradient descent
    current_pos = (x0, y0, f(x0, y0))
    for _ in range(max_iterations):
        tau = 1 
        df = f_grad(x0, y0)
        # Backtracking line search algorithm (goal -> find the best tau!), we use armijo condition here
        while f(x0-tau*f_grad(x0,y0)[0], y0-tau*f_grad(x0,y0)[1]) - f(x0,y0) > -eps*norm(df[0], df[1])**2:
            tau = beta*tau 
        
        # Gradient descent 
        X_grad, Y_grad = f_grad(x0, y0) 
        x0 = x0 - tau * X_grad
        y0 = y0 - tau * Y_grad
        current_pos = (x0, y0, f(x0, y0))
       
        if plot:
            # Plotting 
            ax.plot_surface(X, Y, Z, cmap="viridis", zorder=0)  
            ax.scatter(current_pos[0], current_pos[1], current_pos[2], color="red", zorder=0)
            plt.pause(0.01)
            ax.clear()

    return (x0, y0)


if __name__ == '__main__':
    print("found optimum (without line search) = ", optimize_without_backtracking(1, 1, 0.1, 100, True))
    print("found optimum (with line search) = ", optimize_with_backtracking(1, 1, 1e-4, 0.8, 100, True))
