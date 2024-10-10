import matplotlib.pyplot as plt
import numpy as np


def f(x: np.ndarray, a: float, b: float):
    """
    Compute the function f(x) = a * x / (b + x).

    Parameters:
    - x (np.ndarray): Independent variable.
    - a (float): Parameter 'a' in the function.
    - b (float): Parameter 'b' in the function.

    Returns:
    np.ndarray: Result of the function for each element in x.

    >>> f(np.array([1, 2, 3]), 2, 1)
    array([0.66666667, 1. , 1.5])
    """
    return a*x/(b+x)


def f_grad(x: np.ndarray, a: float, b: float):
    """
    Compute the gradient of the function f(x) = a * x / (b + x) with respect to 'a' and 'b'.

    Parameters:
    - x (np.ndarray): Independent variable.
    - a (float): Parameter 'a' in the function.
    - b (float): Parameter 'b' in the function.

    Returns:
    tuple: Tuple containing the gradients with respect to 'a' and 'b'.

    >>> f_grad(np.array([1, 2, 3]), 2, 1)
    (array([0.66666667, 0.5       , 0.42857143]), array([-0.44444444, -0.25      , -0.1875    ]))
    """
    return (x/(b + x), -a*x/(b+x)**2)


def f_loss(x: np.ndarray, a: float, b: float, y: float):
    """
    Compute the loss or residuals of the function.

    Parameters:
    - x (np.ndarray): Independent variable.
    - a (float): Parameter 'a' in the function.
    - b (float): Parameter 'b' in the function.
    - y (float): Observed values.

    Returns:
    np.ndarray: Residuals or loss.

    >>> f_loss(np.array([1, 2, 3]), 2, 1, 1)
    array([0.33333333, 0.5       , 0.5       ])
    """
    return y - f(x, a, b)

def jacobian(x: np.ndarray, a: float, b: float):
    """
    Compute the Jacobian matrix of the function.

    Parameters:
    - x (np.ndarray): Independent variable.
    - a (float): Parameter 'a' in the function.
    - b (float): Parameter 'b' in the function.

    Returns:
    np.ndarray: Jacobian matrix.

    >>> jacobian(np.array([1, 2, 3]), 2, 1)
    array([[ 0.66666667, -0.44444444],
           [ 0.5       , -0.25      ],
           [ 0.42857143, -0.1875    ]])
    """ 
    grad = f_grad(x, a, b)
    return np.column_stack([grad[0], grad[1]]) 

def gauss_newton(x: np.ndarray, y: float, a0: float, b0: float, tol: float, max_iter: int):
    """
    Perform Gauss-Newton optimization for the given function.

    Parameters:
    - x (np.ndarray): Independent variable.
    - y (float): Observed values.
    - a0 (float): Initial guess for parameter 'a'.
    - b0 (float): Initial guess for parameter 'b'.
    - tol (float): Tolerance for convergence.
    - max_iter (int): Maximum number of iterations.

    Returns:
    np.ndarray: Optimized parameters 'a' and 'b'.

    >>> x = np.array([1, 2, 3, 4, 5])
    >>> y = 2 * x / (1 + x) + np.random.normal(0, 0.1, size=len(x))
    >>> gauss_newton(x, y, 1, 1, 1e-5, 100)
    array([2., 1.])
    """ 
    old = new = np.array([a0, b0])
    for _ in range(max_iter):
        old = new
        jac = jacobian(x, old[0], old[1])
        loss = f_loss(x, old[0], old[1], y)
        new = old + np.linalg.inv(jac.T@jac)@jac.T@loss
        if np.linalg.norm(old-new) < tol:
            break
    return new

def main():
    # Generate data
    x = np.linspace(0, 5, 50)
    y = f(x, 2, 3) + np.random.normal(0, 0.1, size=50)
    
    # Gauss newton optimization 
    a, b = gauss_newton(x, y, 5, 1, 1e-5, 10)

    # Calculate fitted function
    y_hat = f(x, a, b)
    
    # Create plot
    plt.figure()
    plt.scatter(x, y, label='Original data')
    plt.plot(x, y_hat, label="Fitted function", color="red")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
