# Zadanie 1. (7p)
# Zaimplementuj algorytm gradientu prostego oraz algorytm Newtona.
# Algorytm Newtona powinien móc działać w dwóch trybach:
#       ze stałym parametrem kroku
#       z adaptacją parametru kroku przy użyciu metody z nawrotami.
#           w iteracyjny sposób znaleźć

import numpy as np
import numdifftools as ndt
import time
from matplotlib import pyplot as plt
from random import randint

from scipy.special.orthogonal import jacobi


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

    while any(i>error_margin for i in diff) and num_of_iterations<max_num_of_iter:
        prev_x = x
        d = ndt.Gradient(func)(prev_x, alpha, n)
        x = [prev_x[i] - step*d[i] for i in range(len(x))]
        diff = [abs(prev_x[i]-x[i]) for i in range(len(x))]

        num_of_iterations += 1
        y.append(func(x, alpha, n))

    return (x, y, num_of_iterations)


def newtonConstStep(x0, func, step, alpha, n, error_margin=0.0001, max_num_of_iter=1000):
    x = x0
    diff = x0

    num_of_iterations = 0
    y = []

    while any(i>error_margin for i in diff) and num_of_iterations<max_num_of_iter:
        prev_x = x

        # d = hess(x)**(-1)*grad(x)
        d = np.dot(np.linalg.inv(ndt.Hessian(f)(prev_x, alpha, n)), ndt.Gradient(func)(prev_x, alpha, n))

        x = [prev_x[i] - step*d[i] for i in range(len(x))]
        diff = [abs(prev_x[i]-x[i]) for i in range(len(x))]

        num_of_iterations += 1
        y.append(func(x, alpha, n))

    return (x, y, num_of_iterations)


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
    while any(i>error_margin for i in diff) and num_of_iterations<max_num_of_iter:
        prev_x = x

        # d = hess(x)**(-1)*grad(x)
        d = np.dot(np.linalg.inv(ndt.Hessian(f)(prev_x, alpha, n)), ndt.Gradient(func)(prev_x, alpha, n))

        x = [prev_x[i] - t*step*d[i] for i in range(len(x))]
        diff = [abs(prev_x[i]-x[i]) for i in range(len(x))]

        num_of_iterations += 1
        y.append(func(x, alpha, n))

    return (x, y, num_of_iterations)


def localMinimum(algorithm, x0, func, step, alpha, n, betha=1, gamma=1/2, error_margin=0.0001, max_num_of_iter=1000):
    x = x0
    diff = x0
    # for making sure that gradient works properly
    num_of_iter = 0
    next_x = [x] # saves all x positions of algorith
    next_y = [func(x, alpha, n)] # saves all values for next_x
    t = 1

    prev_hess = np.linalg.inv(ndt.Hessian(f)(x, alpha, n))

    if algorithm == 'newtonAdaptStep':
        v = -np.dot(np.linalg.inv(ndt.Hessian(f)(x, alpha, n)), ndt.Gradient(func)(x, alpha, n))
        while (func(x+t*v, alpha, n)>func(x, alpha, n)+gamma*t*np.transpose(ndt.Gradient(func)(x, alpha, n))*v).any(): # tu any czy all???
            t = betha*t

    while any(i>error_margin for i in diff) and num_of_iter<max_num_of_iter:
        prev_x = x

        if algorithm=='gradientDescend' :
            d = ndt.Gradient(func)(prev_x, alpha, n)
        else:
            d = np.asarray(np.dot(prev_hess, ndt.Gradient(func)(prev_x, alpha, n))).reshape(-1)
            #d = np.dot(np.linalg.inv(ndt.Hessian(f)(prev_x, alpha, n)), ndt.Gradient(func)(prev_x, alpha, n))

        x = [prev_x[i] - t*step*d[i] for i in range(len(x))]
        diff = [abs(prev_x[i]-x[i]) for i in range(len(x))]

        num_of_iter += 1
        next_y.append(func(x, alpha, n))
        next_x.append(x)
        if not algorithm=='gradientDescend' :
            x_diff = np.transpose(np.matrix(np.subtract(prev_x, x)))
            grad_diff = np.transpose(np.matrix(np.subtract(ndt.Gradient(func)(prev_x, alpha, n), ndt.Gradient(func)(x, alpha, n))))
            prev_hess = inv_hess(prev_hess, x_diff, grad_diff)
    return (x, next_x, next_y, num_of_iter)


def f(x, alpha, n):
    result = 0
    for i in range(n):
        result += alpha**(i/(n-1))*x[i]**2
    return result

# DFP
def inv_hess(prev_inv_hess, x_diff, grad_diff):
    x_diff_t = np.transpose(x_diff)

    grad_diff_t = np.transpose(grad_diff)
    a = x_diff@x_diff_t
    c=(x_diff_t*grad_diff)
    b = prev_inv_hess*grad_diff*grad_diff_t*prev_inv_hess/(grad_diff_t*prev_inv_hess*grad_diff)
    return prev_inv_hess+x_diff*x_diff_t/(x_diff_t*grad_diff)-prev_inv_hess*grad_diff*grad_diff_t*prev_inv_hess/(grad_diff_t*prev_inv_hess*grad_diff)


def iterToStep(algorithm, fig, alpha, step, n, x0):
    if n == 10:
        k = 0
    else:
        k = 3
    for j in range(len(alpha)):
        iterations = []
        for i in range(1, len(step-2)):
            f_min, next_x, next_y, num_of_iter = localMinimum(algorithm, x0, f, step[i], alpha[j], n)
            iterations.append(num_of_iter)
        ax = fig.add_subplot(2, 3, j+k+1)
        plt.plot(step[1:], iterations)
        min_id = iterations.index(min(iterations))
        plt.plot(step[min_id], iterations[min_id], 'y*')
        ax.tick_params(labelsize=7)
        ax.title.set_text(f"alpha={alpha[j]}, n={n}")
        ax.title.set_size(7)
        plt.xlabel("step", fontsize=6)
        plt.ylabel("iterations", fontsize=6)



def main():

    alpha = [1, 10, 100]
    n = [10, 20]
    step = np.linspace(0, 2, 30)
    x0 = np.array([randint(1, 100) for i in range(10)])
    x1 = np.array([randint(1, 100) for i in range(20)])

    # time
    # t_start = time.process_time()
    # min, next_x, next_y, num_of_iter = localMinimum('gradientDescend', x0, f, 0.3, alpha[0], n[0])
    # t_stop = time.process_time()
    # print(f"gradient process time:{t_stop-t_start}")
    # print(num_of_iter)
    # print(min)

    # t_start = time.process_time()
    # min, next_x, next_y, num_of_iter = localMinimum('newtonConstStep', x0, f, 0.9, alpha[0], n[0])
    # t_stop = time.process_time()
    # print(f"newton with constant step process time:{t_stop-t_start}")
    # print(num_of_iter)
    # print(min)

    # t_start = time.process_time()
    # min, next_x, next_y, num_of_iter = localMinimum('newtonAdaptStep', x0, f, 0.9, alpha[0], n[0])
    # t_stop = time.process_time()
    # print(f"newton with backtracking process time:{t_stop-t_start}")
    # print(num_of_iter)
    # print(min)

    #plotting
    fig = plt.figure()
    iterToStep('gradientDescend', fig, alpha, step, 10, x0)
    iterToStep('gradientDescend', fig, alpha, step, 20, x1)


        # x = np.arange(0, 10)
        # y = [f([x[j]]*10, 1, 10) for j in range(len(x))]
        # plt.plot(x, y)
        # plt.plot(min[0], f(min, alpha[0], n[0]), 'g*')
        # plt.plot(next_x, next_y, 'r')
        # plt.plot(range(num_of_iter+1), next_y)

        # ax.title.set_text(f"step={step[i]}")
        # ax.title.set_fontsize(7)
        # ax.tick_params(labelsize=7)


    plt.show()


if __name__=="__main__":
    main()

