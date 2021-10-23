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


# given function
def f(x, alpha, n):
    result = 0
    for i in range(n):
        result += alpha**(i/(n-1))*x[i]**(2)
    return result


gradient =  ndt.Gradient(f)
hessian = ndt.Hessian(f)

def gradientDescend(x0, func, step, alpha, n, error_margin=0.0000001, max_num_of_iter=1000):
    x = x0
    diff = x0
    num_of_iterations = 0
    y = [func(x0, alpha, n)]

    # stop criterion:
    while np.linalg.norm(diff)>error_margin and num_of_iterations<max_num_of_iter:
        prev_x = x
        d = gradient(prev_x, alpha, n)
        x = prev_x - step*d

        if any(abs(x))>100:
            return (x, y, num_of_iterations)

        diff = prev_x-x

        num_of_iterations += 1
        y.append(func(x, alpha, n))

    return (x, y, num_of_iterations)


def newton(x0, func, step, alpha, n, error_margin=0.0000001, max_num_of_iter=1000):
    x = x0
    diff = x0
    num_of_iterations = 0
    y = [func(x0, alpha, n)]
    # estymacja hessjanu:
    #prev_inv_hess = np.linalg.inv(np.eye(n))

    while np.linalg.norm(diff)>error_margin and num_of_iterations<max_num_of_iter:
        prev_x = x
        # d : hess(x)**(-1)*grad(x)
        d = np.linalg.inv(hessian(prev_x, alpha, n))@gradient(prev_x, alpha, n)
        # estymacja hesjanu:
        #d = np.asarray(prev_inv_hess@gradient(x, alpha, n)).reshape(-1)

        x = prev_x - step*d
        diff = prev_x-x

        num_of_iterations += 1
        y.append(func(x, alpha, n))

        # dla estymacji hesjanu
        # x_diff = np.transpose(np.matrix(np.subtract(prev_x, x)))
        # grad_diff = np.transpose(np.matrix(np.subtract(gradient(prev_x, alpha, n), gradient(x, alpha, n))))
        # prev_inv_hess = inv_hess(prev_inv_hess, x_diff, grad_diff)

    return (x, y, num_of_iterations)


def newtonBtr(x0, func, step, alpha, n, betha=0.95, gamma=0.5, error_margin=0.0000001, max_num_of_iter=1000):
    x = x0
    diff = x0
    prev_x = x
    num_of_iterations = 0
    y = [func(x0, alpha, n)]
    # estymacja hessjanu:
    #prev_inv_hess = np.linalg.inv(np.eye(n))
    t = 1
    while np.linalg.norm(diff)>error_margin and num_of_iterations<max_num_of_iter:
        prev_x = x

        # d : hess(x)**(-1)*grad(x)
        grad = gradient(prev_x, alpha, n)
        d = np.linalg.inv(hessian(prev_x, alpha, n))@grad
        # estymacja hesjanu:
        #d = np.asarray(prev_inv_hess@gradient(x, alpha, n)).reshape(-1)

        x = prev_x - step*t*d
        diff = prev_x-x

        num_of_iterations += 1
        y.append(func(x, alpha, n))

        #TODO tutaj while czy if, i czy to przed czy po
        if func(x, alpha, n)>func(prev_x, alpha, n)+gamma*t*np.transpose(grad)@(-d):
            t = t*betha

        # dla estymacji hesjanu
        # x_diff = np.transpose(np.matrix(np.subtract(prev_x, x)))
        # grad_diff = np.transpose(np.matrix(np.subtract(gradient(prev_x, alpha, n), gradient(x, alpha, n))))
        # prev_inv_hess = inv_hess(prev_inv_hess, x_diff, grad_diff)

    return (x, y, num_of_iterations)


# DFP inverted hessian estimation
def inv_hess(prev_inv_hess, x_diff, grad_diff):
    x_diff_t = np.transpose(x_diff)
    grad_diff_t = np.transpose(grad_diff)

    return prev_inv_hess+x_diff@x_diff_t/(x_diff_t@grad_diff)-(prev_inv_hess@grad_diff@grad_diff_t@prev_inv_hess)/(grad_diff_t@prev_inv_hess@grad_diff)
