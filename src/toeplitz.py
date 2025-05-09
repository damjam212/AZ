import numpy as np
from .fft import fft, ifft
from .utils import pad_to_power_of_two

def toeplitz_matvec(t_col: np.ndarray, t_row: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Mnoży macierz Toeplitza T (zdefiniowaną przez t_col i t_row) przez wektor x w O(n log n).

    t_col: pierwsza kolumna T (długość n)
    t_row: pierwszy wiersz T (długość n), t_row[0] == t_col[0]
    x:     wektor wejściowy (długość n)
    zwraca: y = T @ x
    """
    n = x.shape[0]
    assert t_col.shape[0] == n and t_row.shape[0] == n
    assert t_col[0] == t_row[0]

    # Konstruowanie rozszerzonych wektorów długości 2n-1
    c = np.concatenate([t_row, np.array([t_col[0]]), np.flip(t_col[1:])])  # długość 2n-1
    x_ext = np.concatenate([x, np.zeros(n-1)])   # długość 2n-1
    
    # Dopasowanie do potęgi dwójki
    m = pad_to_power_of_two(c)
    c_padded = pad_to_power_of_two(c)
    x_padded = pad_to_power_of_two(x_ext)

    # FFT
    C = fft(c_padded)
    X = fft(x_padded)

    # Punktowy iloczyn i IFFT
    y_full = ifft(C * X)

    # Pobranie korelacyjnej części (indeksy n-1 do 2n-2)
    result = y_full[0:n].real
    
    return result