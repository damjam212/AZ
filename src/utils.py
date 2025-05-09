import numpy as np

def pad_to_power_of_two(x: np.ndarray) -> np.ndarray:
    """
    Rozszerza wektor x do najbliższej wyższej potęgi dwójki, dopełniając zerami.
    """
    n = x.shape[0]
    m = 1 << (n-1).bit_length()
    if m == n:
        return x
    return np.concatenate([x, np.zeros(m - n, dtype=x.dtype)])
