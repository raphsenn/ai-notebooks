import pytest
from x_squared import optimize_without_backtracking, optimize_with_backtracking


def test_without_backtracking():
    assert optimize_without_backtracking(8, 0.09, 100, False) < 1e-7


def test_with_backtracking():
    assert optimize_with_backtracking(8, 1e-4, 0.9, 100, False) < 1e-8
