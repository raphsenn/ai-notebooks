import pytest
from xy_squared import optimize_without_backtracking, optimize_with_backtracking


def test_without_backtracking():
    optimum = optimize_without_backtracking(1, 1, 0.1, 100, False)
    assert optimum[0] < 1e-9
    assert optimum[1] < 1e-9

def test_with_backtracking():
    optimum = optimize_with_backtracking(1, 1, 1e-4, 0.8, 100, False)
    assert optimum[0] < 1e-22
    assert optimum[1] < 1e-22
