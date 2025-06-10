import numpy as np

def generate_input_file(n, filename, min_val, max_val):
    try:
        n = int(n)
        if n <= 0:
            raise ValueError("Rozmiar n musi być dodatni.")
    except ValueError:
        print("Nieprawidłowy rozmiar n. Podaj liczbę całkowitą dodatnią.")
        return

    try:
        min_val = int(min_val)
        max_val = int(max_val)
        if min_val > max_val:
            raise ValueError("Dolna granica musi być mniejsza lub równa górnej granicy.")
    except ValueError:
        print("Nieprawidłowy zakres wartości. Podaj liczby całkowite, gdzie dolna granica <= górna granica.")
        return

    # Generowanie losowych danych
    t_col = np.random.randint(min_val, max_val + 1, size=n).astype(float)
    t_row = np.concatenate(([t_col[0]], np.random.randint(min_val, max_val + 1, size=n-1))).astype(float)
    x = np.random.randint(min_val, max_val + 1, size=n).astype(float)

    # Zapisywanie do pliku
    try:
        with open(filename, 'w') as f:
            f.write("1\n")  # Jedna instancja
            f.write(f"{n}\n")
            f.write(' '.join(f'{val:.12f}' for val in t_col) + '\n')
            f.write(' '.join(f'{val:.12f}' for val in t_row) + '\n')
            f.write(' '.join(f'{val:.12f}' for val in x) + '\n')
        print(f"Plik wejściowy '{filename}' został wygenerowany pomyślnie.")
    except Exception as e:
        print(f"Błąd podczas zapisywania pliku: {e}")

def main():
    print("--- Generator pliku wejściowego ---")
    n = input("Podaj rozmiar n: ")
    min_val = input("Podaj dolną granicę zakresu wartości (całkowita): ")
    max_val = input("Podaj górną granicę zakresu wartości (całkowita): ")
    filename = input("Podaj nazwę pliku (np. test.txt): ")
    if not filename.endswith('.txt'):
        filename += '.txt'
    generate_input_file(n, filename, min_val, max_val)

if __name__ == '__main__':
    main()