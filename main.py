import os
import time
import numpy as np
import matplotlib.pyplot as plt
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

def read_generation_time(filename):
    try:
        with open(filename, 'r') as f:
            line = f.readline()
            time_str = line.strip().split()[0]
            return float(time_str)
    except FileNotFoundError:
        print(f"NIE ZNALEZIONO PLIKU TIME  {filename}")
        return None

GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'

def compare_results(computed, expected, atol=1e-10):
    if len(computed) != len(expected):
        print(f"{RED}[NO]{RESET} Liczba przypadków testowych nie zgadza się.")
        return False

    all_passed = True
    for i, (y_comp, y_exp) in enumerate(zip(computed, expected)):
        if np.allclose(y_comp, y_exp, atol=atol):
            print(f"{GREEN}[OK]{RESET} Test {i + 1} zakończony sukcesem.")
        else:
            print(f"{RED}[NO]{RESET} Test {i + 1} nie powiódł się.")
            all_passed = False
    return all_passed
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def plot_time_complexity(execution_data):
    if not execution_data:
        print("Brak danych do wykresu.")
        return

    # Grupowanie i uśrednianie
    grouped = defaultdict(list)
    for n, t in execution_data:
        grouped[n].append(t)

    ns = sorted(grouped.keys())
    avg_times = np.array([np.mean(grouped[n]) for n in ns])

    # Teoretyczna złożoność: O(n log n)
    n_log_n = np.array(ns) * np.log2(ns)
    n_log_n = n_log_n / n_log_n.max() * avg_times.max()  # przeskalowanie

    # Teoretyczna złożoność: O(n)
    linear = np.array(ns)
    linear = linear / linear.max() * avg_times.max()  # przeskalowanie

    # Wykres
    plt.figure(figsize=(10, 6))
    plt.plot(ns, avg_times, 'o-', label='Średni rzeczywisty czas wykonania')
    plt.plot(ns, n_log_n, '--', label='Złożoność teoretyczna: O(n log n)')
    plt.plot(ns, linear, ':', label='Złożoność liniowa(punkt odniesienia): O(n)')
    plt.xlabel('Rozmiar wejścia n')
    plt.ylabel('Czas (s)')
    plt.title('Charakterystyka czasowa algorytmu toeplitz_matvec')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    input_dir = 'data/input'
    output_dir = 'data/output'
    time_dir = 'data/time'
    
    input_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    
    total_tests = 0
    total_passed = 0
    total_failed = 0
    execution_data = []  # lista (n, czas)

    for input_file in input_files:
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
        input_path = os.path.join(input_dir, input_file)
        output_file = input_file.replace('input', 'output')
        output_path = os.path.join(output_dir, output_file)
        time_file = input_file.replace('input', 'time')
        time_path = os.path.join(time_dir, time_file)
        
        if not os.path.exists(output_path):
            print(f'Brak odpowiadającego pliku wyjściowego dla {input_file}')
            continue
        
        cases = read_input_file(input_path)
        expected_results = read_output_file(output_path)
        generation_time = read_generation_time(time_path)
        
        computed_results = []
        start_time = time.perf_counter()

        for n, t_col, t_row, x in cases:
            start_case = time.perf_counter()
            y = toeplitz_matvec(t_col, t_row, x)
            end_case = time.perf_counter()
            computed_results.append(y)
            execution_data.append((n, end_case - start_case))

        end_time = time.perf_counter()
        execution_time = end_time - start_time
        
        print(f"\nTest pliku: {input_file}")
        if compare_results(computed_results, expected_results):
            print(f'Test {input_file} zakończony sukcesem.')
            total_passed += 1
        else:
            print(f'Test {input_file} nie powiódł się.')
            total_failed += 1
        total_tests += 1
        
        print(f"Czas wykonania toeplitz_matvec: {execution_time:.6f} sekund")
        if generation_time is not None:
            print(f"Czas generowania danych: {generation_time:.6f} sekund")
        else:
            print("Brak informacji o czasie generowania danych.")
            
        print("\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    
    print("\n==================== Podsumowanie testów ====================")
    print(f"Całkowita liczba testów: {total_tests}")
    print(f"Liczba testów zakończonych sukcesem: {total_passed}  {GREEN}[OK]{RESET}")
    print(f"Liczba testów zakończonych niepowodzeniem: {total_failed}  {RED}[NO]{RESET}")
    print("=============================================================")

    # Rysuj wykres czasów
    plot_time_complexity(execution_data)

if __name__ == '__main__':
    main()
