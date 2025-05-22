import numpy as np
import pytest
from src.toeplitz import toeplitz_matvec


def naive_toeplitz_prod(t_col, t_row, x):
    from scipy.linalg import toeplitz
    T = toeplitz(t_row, t_col)
    
    return T @ x


@pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 32])
def test_toeplitz_vs_naive(n):
    np.random.seed(42)
    t_col = np.random.randn(n)
    t_row = np.concatenate(([t_col[0]], np.random.randn(n-1)))
    x = np.random.randn(n)
    y_fast = toeplitz_matvec(t_col, t_row, x)
    y_naive = naive_toeplitz_prod(t_col, t_row, x)
    assert np.allclose(y_fast, y_naive, atol=1e-10)


def test_toeplitz_zero_vector():
    n = 8
    t_col = np.random.randn(n)
    t_row = np.concatenate(([t_col[0]], np.random.randn(n-1)))
    x = np.zeros(n)
    y_fast = toeplitz_matvec(t_col, t_row, x)
    assert np.allclose(y_fast, np.zeros(n))


def test_toeplitz_identity():
    # Toeplitz z t_col = t_row = [1, 0, 0, ..., 0] jest macierzą jednostkową
    n = 8
    t_col = np.zeros(n)
    t_col[0] = 1.0
    t_row = np.zeros(n)
    t_row[0] = 1.0
    x = np.random.randn(n)
    y_fast = toeplitz_matvec(t_col, t_row, x)
    assert np.allclose(y_fast, x)


def test_toeplitz_size_one():
    t_col = np.array([3.14])
    t_row = np.array([3.14])
    x = np.array([2.0])
    y_fast = toeplitz_matvec(t_col, t_row, x)
    assert np.allclose(y_fast, np.array([3.14*2.0]))


def test_toeplitz_non_power_of_two():
    n = 7
    t_col = np.random.randn(n)
    t_row = np.concatenate(([t_col[0]], np.random.randn(n-1)))
    x = np.random.randn(n)
    y_fast = toeplitz_matvec(t_col, t_row, x)
    y_naive = naive_toeplitz_prod(t_col, t_row, x)

    print(f'y_fast {y_fast}')
    print(f'y_naive {y_naive}')
    assert np.allclose(y_fast, y_naive, atol=1e-10)
