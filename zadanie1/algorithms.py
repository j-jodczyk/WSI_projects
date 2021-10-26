# Zadanie 1. (7p)
# Zaimplementuj algorytm gradientu prostego oraz algorytm Newtona.
# Algorytm Newtona powinien móc działać w dwóch trybach:
#       ze stałym parametrem kroku
#       z adaptacją parametru kroku przy użyciu metody z nawrotami.
#           w iteracyjny sposób znaleźć

import numpy as np
import numdifftools as ndt
from matplotlib import pyplot as plt

MAX_X = 100


def f(x, alpha, n):
    """
    Objective function.
    """
    result = 0
    for i in range(n):
        result += alpha**(i/(n-1))*x[i]**(2)
    return result


def message(iteration, max_iteration):
    """
    Returns message about why an algorithm finished working.
    """
    return "Algorithm reached iteration limit." if iteration == max_iteration else "Algorithm reached stop criterion."


def gradientDescend(x0, func, step, error_margin=0.0000001, max_iteration=5000):
    """
    Implements gradient descend method of finding minimum od an objective function.

    : param x0: array of begining co-ordinates
    : param func: objective function
    : param step: step of algorithm
    : error_margin: defines the accuracy of stop criterion
    : max_iteration: defines maximum number of iterations that the algorithm can perform
    return: message why the algorithm finished working, co-ordinates of minimal value,
            minimal value, number of iterations, value of function at every iteration (for plotting)
    """
    gradient =  ndt.Gradient(func)

    x = x0
    diff = x0
    iteration = 0
    y = [func(x)]

    # stop criterion:
    while np.linalg.norm(diff)>error_margin and iteration<max_iteration:
        prev_x = x
        d = gradient(prev_x)
        x = prev_x - step*d

        if any(abs(x))>100:
            return ("Divergent function", x, func(x), iteration)

        diff = prev_x-x

        iteration += 1

        y.append(func(x))

    return (message(iteration, max_iteration), x, func(x), iteration, y)


def newton(x0, func, step, error_margin=0.0000001, max_iteration=5000):
    """
    Implements Newton's method of finding minimum od an objective function.

    parameters the same as in gradientDescend
    """
    gradient =  ndt.Gradient(func)
    hessian = ndt.Hessian(func)

    x = x0
    diff = x0
    iteration = 0
    y = [func(x)]

    while np.linalg.norm(diff)>error_margin and iteration<max_iteration:
        prev_x = x
        # d : hess(x)**(-1)*grad(x)
        d = np.linalg.inv(hessian(prev_x))@gradient(prev_x)

        x = prev_x - step*d
        diff = prev_x-x

        iteration += 1
        y.append(func(x))

    return (message(iteration, max_iteration), x, func(x), iteration, y)


def newtonBtr(x0, func, step, betha=0.9, gamma=0.5, error_margin=0.0000001, max_iteration=5000):
    """
    Implements Newton's method with backtracking of finding minimum od an objective function.

    parameters the same as in gradientDescend and:
    : param betha: how much will the step be adjusted
    : param gamma: used in assertion of wheather step needs adjusting
    """
    gradient =  ndt.Gradient(func)
    hessian = ndt.Hessian(func)

    x = x0
    diff = x0
    prev_x = x
    iteration = 0
    y = [func(x)]

    while np.linalg.norm(diff)>error_margin and iteration<max_iteration:
        prev_x = x

        # d : hess(x)**(-1)*grad(x)
        grad = gradient(prev_x)
        d = np.linalg.inv(hessian(prev_x))@grad

        x = prev_x - step*d
        diff = prev_x-x

        while func(x)>func(prev_x)+gamma*step*np.transpose(grad)@(-d):
            step *= betha
            x = prev_x - step*d
            diff = prev_x-x

        iteration += 1
        y.append(func(x))

    return (message(iteration, max_iteration), x, func(x), iteration, y)
