import numpy as np
from matplotlib import pyplot as plt


def f(x: np.ndarray) -> float:
    """
    Represents the function f: R -> R with f(x) = x², where R is the set of real numbers.

    Parameters:
    x (np.ndarray): Input value.

    Returns:
    float: Result of the function f(x) = x².
    """
    return x * x


def f_grad(x: np.ndarray) -> float:
    """
    Represents the gradient function f': R -> R with f'(x) = 2*x.

    Parameters:
    x (np.ndarray): Input value.

    Returns:
    float: Result of the gradient function f'(x) = 2*x.
    """
    return 2 * x


def optimize_without_backtracking(x0: float, tau: float, max_iterations: int, plot: bool) -> float:
    """
    Performs gradient descent optimization without backtracking line search for the function f(x) = x².

    Parameters:
    x0 (float): Initial guess for the optimization.
    tau (float): Learning rate.
    max_iterations (int): Number of maximal iterations.
    plot (bool): Plot flag.

    Returns:
    float: Optimal value of x that minimizes the function f(x) = x².
    """
    # Data for plotting f(x)
    x = np.linspace(-10, 10, 1000)
    y = f(x)

    if plot: 
        # Plotting setup
        plt.figure(num="f(x) = x²")
        plt.xlabel("x")
        plt.ylabel("y")

    # Start gradient descent
    current_pos = (x0, f(x0))
    for _ in range(max_iterations):
        # Update x using gradient descent: x_new = x_old - tau * f_grad(x_old)
        x0 = x0 - tau * f_grad(x0)

        # Update current position
        current_pos = (x0, f(x0))

        if plot: 
            # Plotting
            plt.plot(x, y)
            plt.scatter(current_pos[0], current_pos[1], color="red")
            plt.pause(0.0001)
            plt.clf()

    return x0


def optimize_with_backtracking(x0: float, eps: float, beta: float, max_iterations: int, plot: bool) -> float:
    """
    Performs gradient descent optimization with backtracking line search for the function f(x) = x².

    Parameters:
    x0 (float): Initial guess for the optimization.
    eps (float) Epsilon for armijo condition.
    beta (float): Backtracking parameter.
    plot (bool): Plot flag

    Returns:
    float: Optimal value of x that minimizes the function f(x) = x².
    """
    # Data for plotting f(x)
    x = np.linspace(-10, 10, 1000)
    y = f(x)

    if plot: 
        # Plotting setup
        current_pos = (x0, f(x0))
        plt.figure(num="f(x) = x²")
        plt.xlabel("x")
        plt.ylabel("y")

    # Start gradient descent
    for _ in range(max_iterations):
        tau = 1

        # Backtracking algorithm to find the best tau
        # Use Armijo condition: f(x0 - tau * f_grad(x0)) - f(x0) <= - eps * f_grad(x0) ** 2
        while f(x0 - tau * f_grad(x0)) - f(x0) > -eps * f_grad(x0) ** 2:
            tau = beta * tau

        # Update x using gradient descent: x_new = x_old - tau * f_grad(x_old)
        x0 = x0 - tau * f_grad(x0)

        # Update current position
        current_pos = (x0, f(x0))
        
        if plot:
            # Plotting
            plt.plot(x, y)
            plt.scatter(current_pos[0], current_pos[1], color="red")
            plt.pause(0.0001)
            plt.clf()

    return x0


if __name__ == "__main__":
    print("Found optimum (without line search) =", optimize_without_backtracking(8, 0.09, 100, True))
    print("Found optimum (with line search) =", optimize_with_backtracking(8, 1e-4, 0.9, 100, True))
