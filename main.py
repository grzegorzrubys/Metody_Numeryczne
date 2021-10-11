from math import pi

import numpy as np


def cylinder_area(r: float, h: float):
    """Obliczenie pola powierzchni walca.
    Szczegółowy opis w zadaniu 1.

    Parameters:
    r (float): promień podstawy walca
    h (float): wysokosć walca

    Returns:
    float: pole powierzchni walca
    """
    if r > 0 and h > 0:
        return 2*pi*r*r + 2*pi*r*h
    else:
        return np.NaN


# Zadanie 2
print(np.arange(2, 23, 4))
print(np.arange(5))
print(np.arange(1, 7))
print(np.linspace(2, 4, 10))
print(np.linspace(4, 5, 5, False))
print(np.linspace(4, 5, 5, True))

def fib(n: int):
    """Obliczenie pierwszych n wyrazów ciągu Fibonnaciego.
    Szczegółowy opis w zadaniu 3.

    Parameters:
    n (int): liczba określająca ilość wyrazów ciągu do obliczenia

    Returns:
    np.ndarray: wektor n pierwszych wyrazów ciągu Fibonnaciego.
    """
    lst = [1, 1]
    if n <= 0:
        return None
    if isinstance(n, int):
        if n == 1:
            return np.array([1])
        if n == 2:
            return np.array([1, 1])
        else:
            for i in range(2, n):
                lst.append(lst[i-1] + lst[i - 2])
            fib_result = np.array(lst)
            return np.reshape(fib_result, (1, n))

    return None



def matrix_calculations(a: float):
    """Funkcja zwraca wartości obliczeń na macierzy stworzonej
    na podstawie parametru a.
    Szczegółowy opis w zadaniu 4.

    Parameters:
    a (float): wartość liczbowa

    Returns:
    touple: krotka zawierająca wyniki obliczeń
    (Minv, Mt, Mdet) - opis parametrów w zadaniu 4.
    """
    M = np.array([[a, 1, -a], [0, 1, 1], [-a, a, 1]])
    Mdet = np.linalg.det(M)
    Mt = np.transpose(M)
    if Mdet == 0:
        Minv = np.NaN
    else:
        Minv = np.linalg.inv(M)
    return Minv, Mt, Mdet


#Zadanie 5
M = np.array([[3, 1, -2, 4], [0, 1, 1, 5], [-2, 1, 1, 6], [4, 3, 0, 1]])
print(M[1, 1], M[3, 3], M[3, 2])
w1 = M[:, 2]
w2 = M[1, :]
print(w1, w2)


def custom_matrix(m: int, n: int):
    """Funkcja zwraca macierz o wymiarze mxn zgodnie
    z opisem zadania 7.

    Parameters:
    m (int): ilość wierszy macierzy
    n (int): ilość kolumn macierzy

    Returns:
    np.ndarray: macierz zgodna z opisem z zadania 7.
    """
    if isinstance(m, int) and isinstance(n, int) and m > 0 and n > 0:
        matrix = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                if i <= j:
                    matrix[i, j] = j
                else:
                    matrix[i, j] = i
        return matrix
    else:
        return None


# Zadanie 7


v = np.array([1, 3, 13])
v1 = v.T
v2 = np.array([8, 5, -2])
print(np.multiply(v1, 4))
print(np.array([2, 2, 2]) - v2)
print(np.dot(v1, v2))

# Zadanie 8
M1 = np.array([[1, -7, 3], [-12, 3, 4], [5, 13, -3]])
M2 = np.multiply(M1, 3)
print(M2)
print(M2 + np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]))
print(M1.T)
print(np.dot(M1, v1))
print(np.dot(v2.T, M1))
