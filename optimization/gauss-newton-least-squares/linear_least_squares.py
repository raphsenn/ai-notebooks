import matplotlib.pyplot as plt
import numpy as np


def f(x: np.ndarray, m: float, b: float) -> float:
    """
    Linear function.

    Parameters:
    - x (array-like): Independent variable.
    - m (float): Slope of the line.
    - b (float): Y-intercept of the line.

    Returns:
    float: Result of the linear function.
    """
    return x*m + b


def f_grad(x: np.ndarray, m: float, b: float) -> tuple:
    """
    Gradient of the linear function with respect to parameters m and b.

    Parameters:
    - x (array-like): Independent variable.
    - m (float): Slope of the line.
    - b (float): Y-intercept of the line.

    Returns:
    tuple: Tuple containing the gradients with respect to m and b.
    """
    grad_m = x # df/dm = x
    grad_b = np.ones_like(x) # df/db = 1
    return (grad_m, grad_b)


def f_loss(x: np.ndarray, m: float, b: float, y: float) -> float:
    """
    Residuals or loss of the linear function.

    Parameters:
    - x (array-like): Independent variable.
    - m (float): Slope of the line.
    - b (float): Y-intercept of the line.
    - y (array-like): Observed values.

    Returns:
    array-like: Residuals or loss.
    """
    return y - f(x, m, b)


def jacobian(x: np.ndarray, m: float, b: float):
    """
    Jacobian matrix of the linear function with respect to parameters m and b.

    Parameters:
    - x (array-like): Independent variable.
    - m (float): Slope of the line.
    - b (float): Y-intercept of the line.

    Returns:
    array-like: Jacobian matrix.
    """
    grad = f_grad(x, m, b)
    return np.column_stack([grad[0], grad[1]])


def gauss_newton(x: np.ndarray, y: np.ndarray, m0: float, b0: float, tol: float, max_iter: int):
    """
    Gauss-Newton optimization for linear regression.

    Parameters:
    - x (array-like): Independent variable.
    - y (array-like): Observed values.
    - m0 (float): Initial guess for the slope.
    - b0 (float): Initial guess for the y-intercept.
    - tol (float): Tolerance for convergence.
    - max_iter (int): Maximum number of iterations.

    Returns:
    array-like: Optimized parameters (slope, y-intercept).
    """
    old = np.array([m0, b0])
    for _ in range(max_iter):
        jac = jacobian(x, old[0], old[1])
        loss = f_loss(x, old[0], old[1], y)
        new = old + np.linalg.inv(jac.T@jac)@jac.T@loss
        if np.linalg.norm(old-new) < tol:
            break
    return new


def main():
    # Generate data
    x = np.linspace(0, 5, 50)
    y = f(x, 1, 3) + np.random.normal(0, 0.5, size=50)
    
    # Gauss newton optimization
    m, b = gauss_newton(x, y, 1, 1, 1e-5, 10)
    
    # Calculate fitted function
    y_hat = f(x, m, b)
    
    # Create plot
    plt.figure()
    plt.scatter(x, y, label='Original Data')
    plt.plot(x, y_hat, color="red", label="Fitted function") 
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()


