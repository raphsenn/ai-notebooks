"""
Conjugate Gradient Method for Solving Linear Systems

Consider the linear system Ax = b given by:

    A x = [ 4 1 ] [ x1 ] = [ 1 ] 
          [ 1 3 ] [ x2 ]   [ 2 ] 

where A is a 2x2 matrix, x is the solution vector, and b is the right-hand side vector.

We will perform the Conjugate Gradient Method, starting with an initial guess:

    x_0 = [ 2 ] 
          [ 1 ]

to find an approximate solution to the system.

Usage:
- Define the matrix A, right-hand side vector b, and an initial guess x_0.
- Call the conjugate_gradient_method function with the initial guess, b, and the maximum number of iterations.
- The function will iteratively approximate the solution and display a plot showing the progress.

Example:
    b = np.array([1, 2])
    x_0 = np.array([2, 1])
    print(conjugate_gradient_method(x_0, b, max_iterations=10))
"""
import numpy as np


def A(x: np.ndarray) -> np.ndarray:
    """
    Matrix-vector multiplication function.

    Parameters:
        - x (np.ndarray): Input vector.

    Returns:
        - np.ndarray: Result of the matrix-vector multiplication.
    """
    return np.array([[4, 1], [1, 3]]) @ x


def conjugate_gradient_method(x_0: np.array, b: np.array, max_iterations: int) -> np.ndarray:
    """
    Conjugate Gradient Method to solve the linear system Ax = b.

    Parameters:
        - x_0 (np.ndarray): Initial guess for the solution vector.
        - b (np.ndarray): Right-hand side vector of the linear system.
        - max_iterations (int): Maximum number of iterations.

    Returns:
        - np.ndarray: Approximate solution to the linear system.
    """ 
    x = x_0
    r = b - A(x)
    p = r
    for _ in range(max_iterations):
        alpha = np.dot(r, r) / np.dot(p, A(p))
        x = x + alpha * p
        r_new = r - alpha * A(p)
        beta = np.dot(r_new, r_new) / np.dot(r, r)
        p = r_new + beta * p
        r = r_new
    return x


if __name__ == '__main__':
    b = np.array([1, 2])
    x_0 = np.array([2, 1])
    print(conjugate_gradient_method(x_0, b, 10))
