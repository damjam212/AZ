import numpy as np

def fft(x: np.ndarray) -> np.ndarray:
    """
    Oblicza dyskretną transformatę Fouriera (FFT) wektora x.
    Założenie: len(x) jest potęgą dwójki.
    """
    n = x.shape[0]

    if n == 1:
        return x.copy()
    even = fft(x[::2])
    odd = fft(x[1::2])

    factor = np.exp(-2j * np.pi * np.arange(n) / n)

    return np.concatenate([even + factor[:n//2] * odd,
                           even - factor[:n//2] * odd])

def ifft(X: np.ndarray) -> np.ndarray:
    """
    Oblicza odwrotną FFT (IFFT) wektora X.
    """
    n = X.shape[0]
    x_conj = np.conjugate(X)
    y = fft(x_conj)
    return np.conjugate(y) / n