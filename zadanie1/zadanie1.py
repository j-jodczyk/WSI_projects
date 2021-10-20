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


def localMinimum(algorithm, x0, func, step, alpha, n, betha=0.75, gamma=0.25, error_margin=0.0001, max_num_of_iter=1000):
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


    while np.linalg.norm(diff)>error_margin and num_of_iter<max_num_of_iter:
        a = np.linalg.norm(diff)
        prev_x = x

        if algorithm=='gradientDescend' :
            d = ndt.Gradient(func)(prev_x, alpha, n)
        else:
            d = np.asarray(np.dot(prev_hess, ndt.Gradient(func)(prev_x, alpha, n))).reshape(-1)
            #d = np.dot(np.linalg.inv(ndt.Hessian(f)(prev_x, alpha, n)), ndt.Gradient(func)(prev_x, alpha, n))

        x = prev_x - t*step*d
        b = np.linalg.norm(d)

        # zadana dziedzina x- czy jest na to potrzeba?
        # for x_i in x:
        #     if x_i>100:
        #         x_i = 100
        #     elif x_i<-100:
        #         x_i = -100
        diff = prev_x-x

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
    min_step = []
    k = 0 if n==10 else 3
    for j in range(len(alpha)):
        iterations = []
        min_y = []
        for i in range(1, len(step-2)):
            f_min, next_x, next_y, num_of_iter = localMinimum(algorithm, x0, f, step[i], alpha[j], n)
            iterations.append(num_of_iter)
            min_y.append(next_y[-1])
        ax = fig.add_subplot(2, 3, j+k+1)
        # plt.plot(step[1:], iterations) # for which step we iterate the least
        plt.plot(step[1:], min_y) # for which step do we get the smallest f
        min_id = iterations.index(min(iterations))
        plt.plot(step[min_id], iterations[min_id], 'y*')
        min_step.append((step[min_id], iterations[min_id], f_min))
        ax.tick_params(labelsize=7)
        ax.title.set_text(f"alpha={alpha[j]}, n={n}")
        ax.title.set_size(7)
        plt.xlabel("step", fontsize=6)
        plt.ylabel("iterations", fontsize=6)
    return min_step


def iterToMin(algorithm, fig, alpha, step, n, x0):
    k = 0 if n==10 else 3
    color = ['g', 'b', 'r']
    for j in range(len(alpha)):
        ax = fig.add_subplot(2, 3, j+k+1)
        for i in range(len(step[j])):
            f_min, next_x, next_y, num_of_iter = localMinimum(algorithm, x0, f, step[j][i], alpha[j], n)
            plt.plot(range(num_of_iter+1), next_y, f'{color[i]}')



def main():

    alpha = [1, 10, 100]
    n = [10, 20]
    step = np.linspace(0, 1, 1000) #potrzebe nowe wykresy
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
    # fig = plt.figure()
    # min_step = iterToStep('gradientDescend', fig, alpha, step, 10, x0)
    # min_step += iterToStep('gradientDescend', fig, alpha, step, 20, x1)
    # print(min_step)

    # fig2 = plt.figure()
    # min_step = iterToStep('newtonConstStep', fig2, alpha, step, 10, x0)
    # min_step += iterToStep('newtonConstStep', fig2, alpha, step, 20, x1)
    # print(min_step)

    # fig3 = plt.figure()
    # min_step = iterToStep('newtonAdaptStep', fig3, alpha, step, 10, x0)
    # min_step += iterToStep('newtonAdaptStep', fig3, alpha, step, 20, x1)
    # print(min_step)

    fig4 = plt.figure()
    iterToMin('gradientDescend', fig4, alpha, [[0.25, 0.4, 0.9], [0.03, 0.02, 0.07], [0.01, 0.001, 0.0001]], 10, x0)
    iterToMin('gradientDescend', fig4, alpha, [[0.25, 0.4, 0.9], [0.03, 0.02, 0.07], [0.01, 0.001, 0.0001]], 20, x1)
    plt.show()
    #plt.savefig("iteration_to_step")


if __name__=="__main__":
    main()

