import os
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from src.toeplitz import toeplitz_matvec

# =========================== FUNKCJE POMOCNICZE ===========================

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

def save_output_file(filename, results):
    with open(filename, 'w') as f:
        for y in results:
            f.write(' '.join(f'{val:.12f}' for val in y) + '\n')

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
        print(f"NIE ZNALEZIONO PLIKU TIME {filename}")
        return None

def compare_results(computed, expected, atol=1e-10):
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'
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
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def plot_time_complexity(execution_data):
    if not execution_data:
        print("Brak danych do wykresu.")
        return

    grouped = defaultdict(list)
    for n, t in execution_data:
        grouped[n].append(t)

    ns = np.array(sorted(grouped.keys()))
    avg_times = np.array([np.mean(grouped[n]) for n in ns])

    # Funkcje teoretyczne
    n_log_n = ns * np.log2(ns)
    linear = ns

    # Skalowanie: dopasowanie pierwszego punktu teoretycznych krzywych do avg_times
    scale_nlogn = avg_times[1] / n_log_n[1]
    scale_linear = avg_times[1] / linear[1]

    n_log_n_scaled = n_log_n * scale_nlogn
    linear_scaled = linear * scale_linear

    # Wykres
    plt.figure(figsize=(10, 6))
    plt.plot(ns, avg_times, 'o-', label='Średni czas wykonania')
    plt.plot(ns, n_log_n_scaled, '--', label='O(n log n) – dopasowane do 1. punktu')
    plt.plot(ns, linear_scaled, ':', label='O(n) – dopasowane do 1. punktu')
    plt.xlabel('Rozmiar wejścia n')
    plt.ylabel('Czas (s)')
    plt.title('Charakterystyka czasowa algorytmu toeplitz_matvec')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ============================= MENU I OPCJE ==============================
def option_one():
    print("\n--- OPCJA 1: Wybór pliku z bieżącego katalogu i uruchomienie algorytmu ---\n")
    txt_files = [f for f in os.listdir('.') if f.endswith('.txt')]
    if not txt_files:
        print("Brak plików .txt w bieżącym katalogu.")
        return

    for idx, fname in enumerate(txt_files):
        print(f"[{idx + 1}] {fname}")

    choice = input("Wybierz numer pliku do przetworzenia: ")
    try:
        selected = txt_files[int(choice) - 1]
    except (ValueError, IndexError):
        print("Nieprawidłowy wybór.")
        return

    print(f"\nWybrano plik: {selected}")
    cases = read_input_file(selected)

    results = []
    for i, (n, t_col, t_row, x) in enumerate(cases):
        y = toeplitz_matvec(t_col, t_row, x)
        results.append(y)
        print(f"\n=== Wynik dla przypadku {i + 1} (n = {n}) ===")
        print(" ".join(f"{val:.12f}" for val in y))

    output_filename = selected.replace('.txt', '_output.txt')
    save_output_file(output_filename, results)
    print(f"\nWyniki zapisano do pliku: {output_filename}\n")


def option_two():
    print("\n--- OPCJA 2: Testowanie z katalogów data/input, output, time ---\n")
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'

    input_dir = 'data/input'
    output_dir = 'data/output'
    time_dir = 'data/time'

    input_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    
    total_tests = 0
    total_passed = 0
    total_failed = 0
    execution_data = []

    for input_file in input_files:
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
        input_path = os.path.join(input_dir, input_file)
        output_path = os.path.join(output_dir, input_file.replace('input', 'output'))
        time_path = os.path.join(time_dir, input_file.replace('input', 'time'))

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

    # plot_time_complexity(execution_data)
import numpy as np
import time
import math

def option_three():
    print("\n--- OPCJA 3: Charakterystyka czasowa algorytmu ---\n")

    sizes = [2**i for i in range(0, 20)]  # od 1024 do 4_194_304 (2^10 do 2^21)
    execution_data = []

    for n in sizes:
        t_col = np.random.randint(-10, 11, size=n)
        t_row = np.concatenate(([t_col[0]], np.random.randint(-10, 11, size=n - 1)))
        x = np.random.randint(-10, 11, size=n)

        start = time.perf_counter()
        _ = toeplitz_matvec(t_col, t_row, x)
        end = time.perf_counter()
        elapsed = end - start
        execution_data.append((n, elapsed))
        print(f"n = {n}, czas = {elapsed:.6f} s")

    # Wyświetlanie tabeli z porównaniami
    print("\nTabela porównawcza:")
    print(f"{'n':>12} | {'czas (s)':>10} | {'ratio czasów':>12} | {'ratio n log n':>14}")
    print("-" * 60)
    for i in range(len(execution_data)):
        n = execution_data[i][0]
        czas = execution_data[i][1]

        if i == 0:
            ratio_czas = "-"
            ratio_nlogn = "-"
        else:
            prev_n = execution_data[i-1][0]
            prev_czas = execution_data[i-1][1]

            ratio_czas = czas / prev_czas if prev_czas > 0 else float('inf')

            nlogn = n * math.log2(n)
            prev_nlogn = prev_n * math.log2(prev_n)
            ratio_nlogn = nlogn / prev_nlogn if prev_nlogn > 0 else float('inf')

        if isinstance(ratio_nlogn, str):
            print(f"{n:12d} | {czas:10.6f} | {str(ratio_czas):>12} | {ratio_nlogn:>14}")
        else:
            print(f"{n:12d} | {czas:10.6f} | {ratio_czas:12.6f} | {ratio_nlogn:14.6f}")


    plot_time_complexity(execution_data)

# ============================== MAIN ====================================

def main():
    while True:
        print("\n==================== MENU ====================")
        print("1. Wczytaj plik .txt z katalogu bieżącego i uruchom algorytm")
        print("2. Uruchom testy z katalogów data/")
        print("3. Charakterystyka czasowa algorytmu")
        print("0. Wyjście")
        print("==============================================")
        choice = input("Wybierz opcję: ")

        if choice == '1':
            option_one()
        elif choice == '2':
            option_two()
        elif choice == '3':
            option_three()
        elif choice == '0':
            print("Zamykam program.")
            break
        else:
            print("Nieprawidłowy wybór. Spróbuj ponownie.")

if __name__ == '__main__':
    main()
