import numpy as np
import pytest
from src.fft import fft, ifft

def test_fft_ifft_identity_real():
    x = np.random.rand(8)
    X = fft(x)
    x_rec = ifft(X)
    assert np.allclose(x, x_rec, atol=1e-10)

def test_fft_ifft_identity_complex():
    x = np.random.rand(8) + 1j * np.random.rand(8)
    X = fft(x)
    x_rec = ifft(X)
    assert np.allclose(x, x_rec, atol=1e-10)

def test_fft_against_numpy():
    x = np.random.rand(8)
    X_custom = fft(x)
    X_numpy = np.fft.fft(x)
    assert np.allclose(X_custom, X_numpy, atol=1e-10)

def test_ifft_against_numpy():
    X = np.random.rand(8) + 1j * np.random.rand(8)
    x_custom = ifft(X)
    x_numpy = np.fft.ifft(X)
    assert np.allclose(x_custom, x_numpy, atol=1e-10)

def test_fft_unit_impulse():
    x = np.zeros(8)
    x[0] = 1
    X = fft(x)
    expected = np.ones(8)
    assert np.allclose(X, expected, atol=1e-10)

def test_fft_constant_signal():
    x = np.ones(8)
    X = fft(x)
    expected = np.zeros(8, dtype=complex)
    expected[0] = 8
    assert np.allclose(X, expected, atol=1e-10)

def test_fft_length_one():
    x = np.array([42.0])
    X = fft(x)
    assert np.allclose(X, x, atol=1e-10)
    x_rec = ifft(X)
    assert np.allclose(x_rec, x, atol=1e-10)

def test_fft_non_power_of_two():
    x = np.random.rand(7)
    with pytest.raises(ValueError):
        fft(x)

@pytest.mark.parametrize("n", [64, 128, 256, 512, 1024, 2048])
def test_fft_large_sizes(n):
    x = np.random.rand(n)
    X_custom = fft(x)
    X_numpy = np.fft.fft(x)
    assert np.allclose(X_custom, X_numpy, atol=1e-8)

@pytest.mark.parametrize("n", [64, 128, 256, 512, 1024, 2048])
def test_ifft_large_sizes(n):
    X = np.random.rand(n) + 1j * np.random.rand(n)
    x_custom = ifft(X)
    x_numpy = np.fft.ifft(X)
    assert np.allclose(x_custom, x_numpy, atol=1e-8)
