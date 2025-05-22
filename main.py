import os
import numpy as np
from src.toeplitz import toeplitz_matvec

def read_input_file(filename):
    with open(filename, 'r') as f:
        lines = f.read().split()
        idx = 0
        num_cases = int(lines[idx])
        idx += 1
        cases = []
        for _ in range(num_cases):
            n = int(lines[idx])
            idx += 1
            t_col = np.array([float(lines[idx + i]) for i in range(n)])
            idx += n
            t_row = np.array([float(lines[idx + i]) for i in range(n)])
            idx += n
            x = np.array([float(lines[idx + i]) for i in range(n)])
            idx += n
            cases.append((n, t_col, t_row, x))
        return cases

def read_output_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        results = []
        for line in lines:
            y = np.array([float(val) for val in line.strip().split()])
            results.append(y)
        return results

def compare_results(computed, expected, atol=1e-10):
    if len(computed) != len(expected):
        return False
    for y_comp, y_exp in zip(computed, expected):
        if not np.allclose(y_comp, y_exp, atol=atol):
            return False
    return True

def main():
    input_dir = 'data/input'
    output_dir = 'data/output'
    
    input_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    
    for input_file in input_files:
        input_path = os.path.join(input_dir, input_file)
        output_file = input_file.replace('input', 'output')
        output_path = os.path.join(output_dir, output_file)
        
        if not os.path.exists(output_path):
            print(f'Brak odpowiadającego pliku wyjściowego dla {input_file}')
            continue
        
        cases = read_input_file(input_path)
        expected_results = read_output_file(output_path)
        
        computed_results = []
        for n, t_col, t_row, x in cases:
            y = toeplitz_matvec(t_col, t_row, x)
            computed_results.append(y)
        
        if compare_results(computed_results, expected_results):
            print(f'Test {input_file} zakończony sukcesem.')
        else:
            print(f'Test {input_file} nie powiódł się.')

if __name__ == '__main__':
    main()
