import os
import numpy as np
from scipy.linalg import toeplitz

def generate_random_case(n):
    t_col = np.random.randint(-10, 11, size=n)
    t_row = np.concatenate(([t_col[0]], np.random.randint(-10, 11, size=n - 1)))
    x = np.random.randint(-10, 11, size=n)
    return t_col, t_row, x

def generate_symmetric_case(n):
    base = np.random.randint(-10, 11, size=n)
    t_col = base
    t_row = base
    x = np.random.randint(-10, 11, size=n)
    return t_col, t_row, x

def generate_unit_case(n):
    t_col = np.ones(n, dtype=int)
    t_row = np.ones(n, dtype=int)
    x = np.random.randint(-10, 11, size=n)
    return t_col, t_row, x

def generate_zero_case(n):
    t_col = np.zeros(n, dtype=int)
    t_row = np.zeros(n, dtype=int)
    x = np.random.randint(-10, 11, size=n)
    return t_col, t_row, x

def generate_increasing_case(n):
    t_col = np.arange(1, n + 1)
    t_row = np.concatenate(([t_col[0]], np.arange(2, n + 1)))
    x = np.random.randint(-10, 11, size=n)
    return t_col, t_row, x

def generate_test_case(n, pattern='random'):
    if pattern == 'random':
        return generate_random_case(n)
    elif pattern == 'symmetric':
        return generate_symmetric_case(n)
    elif pattern == 'unit':
        return generate_unit_case(n)
    elif pattern == 'zero':
        return generate_zero_case(n)
    elif pattern == 'increasing':
        return generate_increasing_case(n)
    else:
        raise ValueError(f"Unknown pattern: {pattern}")

def write_test_file(input_path, output_path, test_cases):
    with open(input_path, 'w') as f_in, open(output_path, 'w') as f_out:
        f_in.write(f"{len(test_cases)}\n")
        for t_col, t_row, x in test_cases:
            n = len(x)
            f_in.write(f"{n}\n")
            f_in.write(' '.join(map(str, t_col)) + '\n')
            f_in.write(' '.join(map(str, t_row)) + '\n')
            f_in.write(' '.join(map(str, x)) + '\n')
            T = toeplitz(t_col, t_row)
            y = T @ x
            f_out.write(' '.join(map(str, y)) + '\n')

def main():
    os.makedirs('data/input', exist_ok=True)
    os.makedirs('data/output', exist_ok=True)

    patterns = ['random', 'symmetric', 'unit', 'zero', 'increasing']
    sizes = [4, 8, 16]

    for pattern in patterns:
        for n in sizes:
            test_cases = [generate_test_case(n, pattern)]
            input_filename = f'data/input/input_{pattern}_{n}.txt'
            output_filename = f'data/output/output_{pattern}_{n}.txt'
            write_test_file(input_filename, output_filename, test_cases)

if __name__ == '__main__':
    main()