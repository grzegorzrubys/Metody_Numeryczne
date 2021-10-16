from math import *

import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
import string
import random
from typing import List


#Zadanie 1

def f(x):
    return x*x*x - 3*x


x1 = np.linspace(-1, 1)
x2 = np.linspace(-5, 5)
x3 = np.linspace(0, 5)
plt.figure()
# Wykres 1
plt.subplot(311)
plt.grid(True)
plt.xlabel("x1")
plt.ylabel("y1")
plt.plot(x1, f(x1), label = "x^3 - 3x")
plt.legend()
# Wykres 2
plt.subplot(312)
plt.grid(True)
plt.xlabel("x2")
plt.ylabel("y2")
plt.plot(x2, f(x2), label = "x^3 - 3x")
plt.legend()
# Wykres 3
plt.subplot(313)
plt.grid(True)
plt.xlabel("x3")
plt.ylabel("y3")
plt.plot(x3, f(x3), label = "x^3 - 3x")
plt.legend()
plt.show()

#Zadanie 2
x = np.linspace(-10, 10)
plt.figure()
plt.subplot(311)
plt.xlim(-1, 1)
plt.ylim(-2, 2)
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.plot(x, f(x), label="x^3 - 3x")
plt.subplot(312)
plt.xlim(-10, -1)
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.plot(x, f(x), label="x^3 - 3x")
plt.subplot(313)
plt.xlim(1, 10)
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.plot(x, f(x), label="x^3 - 3x")
#plt.show()

#Zadanie 3


def Q(m, v):
    return m*(v*v)/2


print("Ciepło wynosi: {wynik1} kCal lub {wynik2} J".format(wynik2=Q(2500, 60)*(100/1296), wynik1=Q(2500, 60) * (100/1296)/0.0041855))

v = np.linspace(0, 200)
plt.figure()
plt.subplot(121)
plt.grid(True)
plt.plot(v, Q(3000, v), label="ciepło dla zadanych wartości")
plt.legend()
plt.subplot(122)
plt.grid(True)
plt.semilogy(v, Q(3000, v), label="ciepło dla zadanych wartości w skali logarytmicznej")
plt.legend()
plt.show()


def compare_plot(x1:np.ndarray,y1:np.ndarray,x2:np.ndarray,y2:np.ndarray,
                 xlabel: str,ylabel:str,title:str,label1:str,label2:str):
    """Funkcja służąca do porównywania dwóch wykresów typu plot.
    Szczegółowy opis w zadaniu 3.

    Parameters:
    x1(np.ndarray): wektor wartości osi x dla pierwszego wykresu,
    y1(np.ndarray): wektor wartości osi y dla pierwszego wykresu,
    x2(np.ndarray): wektor wartości osi x dla drugiego wykresu,
    y2(np.ndarray): wektor wartości osi x dla drugiego wykresu,
    xlabel(str): opis osi x,
    ylabel(str): opis osi y,
    title(str): tytuł wykresu ,
    label1(str): nazwa serii z pierwszego wykresu,
    label2(str): nazwa serii z drugiego wykresu.


    Returns:
    matplotlib.pyplot.figure: wykres zbiorów (x1,y1), (x2,y2) zgody z opisem z zadania 3
    """
    if x1.shape != y1.shape or min(x1.shape) == 0 or x2.shape != y2.shape or min(x2.shape) == 0:
        return None
    else:
        plt.grid(True)
        plt.plot(x1, y1, label=label1, linewidth=4)
        plt.plot(x2, y2, label=label2, color="green", linewidth=2)
        plt.legend()
        plt.xlabel = xlabel
        plt.ylabel = ylabel
        plt.title = title
        plt.show()


# Zadanie 5

def f(x):
    return x+2


def g(x):
    return x*x - 2*sin(x) + 3


x = np.linspace(-3, 3)
y1 = np.array([f(i) for i in x])
y2 = np.array([g(i) for i in x])
compare_plot(x,y1, x, y2, 'x', 'y','Rozwiazanie rownania', 'f(x)', 'g(x)')


def parallel_plot(x1:np.ndarray,y1:np.ndarray,x2:np.ndarray,y2:np.ndarray,
                  x1label:str,y1label:str,x2label:str,y2label:str,title:str,orientation:str):
    """Funkcja służąca do stworzenia dwóch wykresów typu plot w konwencji subplot wertykalnie lub chorycontalnie.
    Szczegółowy opis w zadaniu 5.

    Parameters:
    x1(np.ndarray): wektor wartości osi x dla pierwszego wykresu,
    y1(np.ndarray): wektor wartości osi y dla pierwszego wykresu,
    x2(np.ndarray): wektor wartości osi x dla drugiego wykresu,
    y2(np.ndarray): wektor wartości osi x dla drugiego wykresu,
    x1label(str): opis osi x dla pierwszego wykresu,
    y1label(str): opis osi y dla pierwszego wykresu,
    x2label(str): opis osi x dla drugiego wykresu,
    y2label(str): opis osi y dla drugiego wykresu,
    title(str): tytuł wykresu,
    orientation(str): parametr przyjmujący wartość '-' jeżeli subplot ma posiadać dwa wiersze albo '|' jeżeli ma posiadać dwie kolumny.


    Returns:
    matplotlib.pyplot.figure: wykres zbiorów (x1,y1), (x2,y2) zgody z opisem z zadania 5
    """
    if x1.shape != y1.shape or min(x1.shape) == 0 or x2.shape != y2.shape or min(x2.shape) == 0:
        return None
    else:
        if orientation == '-':
            plt.subplot(211)
            plt.grid(True)
            plt.xlabel = x1label
            plt.ylabel = y1label
            plt.plot(x1, y1)
            plt.title = title
            plt.subplot(212)
            plt.grid(True)
            plt.xlabel = x2label
            plt.ylabel = y2label
            plt.plot(x2, y2)
            plt.title = title
            plt.show()
        elif orientation == '|':
            plt.subplot(121)
            plt.grid(True)
            plt.xlabel = x1label
            plt.ylabel = y1label
            plt.plot(x1, y1)
            plt.title = title
            plt.subplot(122)
            plt.grid(True)
            plt.xlabel = x2label
            plt.ylabel = y2label
            plt.plot(x2, y2)
            plt.title = title
            plt.show()
        else:
            return None


# Zadanie 7
a = 0.015
b = 0.2
th = np.array([i for i in range(-100, 101)])
th2 = np.array([i for i in range(-1,2)])
x = np.array([a*exp(b*i)*cos(i) for i in th])
y = np.array([a*exp(b*i)*sin(i) for i in th])
x1 = np.array([a*exp(b*j)*cos(j) for j in th2])
y1 = np.array([a*exp(b*j)*sin(j) for j in th2])
parallel_plot(x, y, x1, y1, 'x', 'y', 'x', 'y', 'Spirala logarytmiczna', '|')


def log_plot(x:np.ndarray,y:np.ndarray,xlabel:str,ylabel:str,title:str,log_axis:str):
    """Funkcja służąca do tworzenia wykresów ze skalami logarytmicznymi.
    Szczegółowy opis w zadaniu 7.

    Parameters:
    x(np.ndarray): wektor wartości osi x,
    y(np.ndarray): wektor wartości osi y,
    xlabel(str): opis osi x,
    ylabel(str): opis osi y,
    title(str): tytuł wykresu ,
    log_axis(str): wartość oznacza:
        - 'x' oznacza skale logarytmiczną na osi x,
        - 'y' oznacza skale logarytmiczną na osi y,
        - 'xy' oznacza skale logarytmiczną na obu osiach.

    Returns:
    matplotlib.pyplot.figure: wykres zbiorów (x,y) zgody z opisem z zadania 7
    """
    if x.shape != y.shape or min(x.shape) == 0:
        return None
    else:
        if log_axis == 'y':
            plt.semilogy(x, y)
            plt.xlabel = xlabel
            plt.ylabel = ylabel
            plt.title = title
            plt.show()
        elif log_axis == 'x':
            plt.plot(x, y)
            plt.xscale("log")
            plt.xlabel = xlabel
            plt.ylabel = ylabel
            plt.title = title
            plt.show()
        elif log_axis == 'xy':
            plt.semilogy(x, y)
            plt.xscale("log")
            plt.xlabel = xlabel
            plt.ylabel = ylabel
            plt.title = title
            plt.show()
        else:
            return None


# Zadanie 9
log_plot(v, Q(3000, v), 'v', 'Q(v)', "Cieplo", 'x')
log_plot(v, Q(3000, v), 'v', 'Q(v)', "Cieplo", 'y')
log_plot(v, Q(3000, v), 'v', 'Q(v)', "Cieplo", 'xy')