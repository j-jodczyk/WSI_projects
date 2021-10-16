# Zadanie 1. (7p)
# Zaimplementuj algorytm gradientu prostego oraz algorytm Newtona.
# Algorytm Newtona powinien móc działać w dwóch trybach:
#       ze stałym parametrem kroku
#       z adaptacją parametru kroku przy użyciu metody z nawrotami.
#           w iteracyjny sposób znaleźć

import numpy as np
import numdifftools as ndt
from matplotlib import pyplot as plt


# usunąć powtarzanie się kodu!! :(((

# chyba działa
def gradientDescend(x0, func, step, alpha, n, error_margin=0.0001, max_num_of_iter=1000):
    # krok stały, podany przez użytkownika
    # x nalezy do D = [-100, 100]^n
    x = x0
    diff = x0
    # for making sure that gradient works properly
    num_of_iterations = 0
    y = []

    while any (i>error_margin for i in diff) and num_of_iterations<max_num_of_iter:
        prev_x = x
        grad = [i*step for i in ndt.Gradient(func)(prev_x, alpha, n)]
        x = [prev_x[i] - grad[i] for i in range(len(x))]
        diff = [abs(prev_x[i]-x[i]) for i in range(len(x))]

        num_of_iterations += 1
        y.append(func(x, alpha, n))

    return (x, y, num_of_iterations)


# chyba działa
def newtonConstStep(x0, func, step, alpha, n, error_margin=0.0001, max_num_of_iter=1000):
    x = x0
    diff = x0

    num_of_iterations = 0
    y = []

    while any (i>error_margin for i in diff) and num_of_iterations<max_num_of_iter:
        prev_x = x

        # d = hess(x)**(-1)*grad(x)
        d = np.dot(np.linalg.inv(ndt.Hessian(f)(prev_x, alpha, n)), ndt.Gradient(func)(prev_x, alpha, n))

        x = [prev_x[i] - step*d[i] for i in range(len(x))]
        diff = [abs(prev_x[i]-x[i]) for i in range(len(x))]

        num_of_iterations += 1
        y.append(func(x, alpha, n))

    return (x, y, num_of_iterations)


# chyba działa
def newtonAdaptStep(x0, func, step, alpha, n, betha, gamma, error_margin=0.0001, max_num_of_iter=1000):
    x = x0
    diff = x0
    prev_x = x
    t = 1
    num_of_iterations = 0
    y = []
    v = -np.dot(np.linalg.inv(ndt.Hessian(f)(prev_x, alpha, n)), ndt.Gradient(func)(prev_x, alpha, n))
    while (func(x+t*v, alpha, n)>func(x, alpha, n)+gamma*t*np.transpose(ndt.Gradient(func)(x, alpha, n))*v).any(): # tu any czy all???
        t = betha*t
    while any (i>error_margin for i in diff) and num_of_iterations<max_num_of_iter:
        prev_x = x

        # d = hess(x)**(-1)*grad(x)
        d = np.dot(np.linalg.inv(ndt.Hessian(f)(prev_x, alpha, n)), ndt.Gradient(func)(prev_x, alpha, n))

        x = [prev_x[i] - step*d[i] for i in range(len(x))]
        diff = [abs(prev_x[i]-x[i]) for i in range(len(x))]

        num_of_iterations += 1
        y.append(func(x, alpha, n))

    return (x, y, num_of_iterations)


def f(x, alpha, n):
    result = 0
    for i in range(n):
        result += alpha**(i/(n-1))*x[i]**2
    return result


def main():
    alpha = [1, 10, 100]
    n = [10, 20]
    x0 = [15]*10
    min, y, num_of_iter = newtonAdaptStep(x0, f, 0.7, alpha[0], n[0], 0.5, 0.25)
    plt.plot(range(num_of_iter), y)
    plt.show()


if __name__=="__main__":
    main()

