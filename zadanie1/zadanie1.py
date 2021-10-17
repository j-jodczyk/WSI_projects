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

    while any(i>error_margin for i in diff) and num_of_iterations<max_num_of_iter:
        prev_x = x
        d = ndt.Gradient(func)(prev_x, alpha, n)
        x = [prev_x[i] - step*d[i] for i in range(len(x))]
        diff = [abs(prev_x[i]-x[i]) for i in range(len(x))]

        num_of_iterations += 1
        y.append(func(x, alpha, n))

    return (x, y, num_of_iterations)


# odwrócony hesian = 0, co w takim razie?
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


# nie dziala odwracanie hesjanu bo = 0
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
    next_y = [f(x, alpha, n)] # saves all values for next_x
    t = 1

    if algorithm == 'newtonAdaptStep':
        v = -np.dot(np.linalg.inv(ndt.Hessian(f)(x, alpha, n)), ndt.Gradient(func)(x, alpha, n))
        while (func(x+t*v, alpha, n)>func(x, alpha, n)+gamma*t*np.transpose(ndt.Gradient(func)(x, alpha, n))*v).any(): # tu any czy all???
            t = betha*t

    while any(i>error_margin for i in diff) and num_of_iter<max_num_of_iter:
        prev_x = x

        if algorithm=='gradientDescend' :
            d = ndt.Gradient(func)(prev_x, alpha, n)
        else:
            d = np.dot(np.linalg.inv(ndt.Hessian(f)(prev_x, alpha, n)), ndt.Gradient(func)(prev_x, alpha, n))

        x = [prev_x[i] - t*step*d[i] for i in range(len(x))]
        diff = [abs(prev_x[i]-x[i]) for i in range(len(x))]

        num_of_iter += 1
        next_y.append(func(x, alpha, n))
        next_x.append(x)
    return (x, next_x, next_y, num_of_iter)


def f(x, alpha, n):
    result = 0
    for i in range(n):
        result += alpha**(i/(n-1))*x[i]**2
    return result

# TODO: fix hessian in Newton
# TODO: measure times
# TODO: measure influence of step
#
def main():
    alpha = [1, 10, 100]
    n = [10, 20]
    step = np.linspace(0, 1, 8)
    x0 = [10]*20
    #for a in alpha:
    #    for b in n:
    fig = plt.figure()
    fig.suptitle("alpha=1, n=20")
    for i in range(1, len(step)-1):
        min, next_x, next_y, num_of_iter = localMinimum('gradientDescend', x0, f, step[i], alpha[0], n[1])
        ax = fig.add_subplot(2, 3, i)
        x = np.arange(0, 10)
        y = [f([x[j]]*10, 1, 10) for j in range(len(x))]
        plt.plot(x, y)
        plt.plot(min[0], f(min, alpha[0], n[1]), 'g*')
        plt.plot(next_x, next_y, 'r')
        ax.title.set_text(f"step={step[i]}")
        ax.title.set_fontsize(7)
        ax.tick_params(labelsize=7)

    #plt.plot(range(num_of_iter), y)

    # x = np.arange(-10, 10)
    # y = [f([x[j]]*10, 1, 10) for j in range(len(x))]
    # plt.plot(x, y)
    # plt.plot(min[1], f(min, 1, 10), 'g*')

    plt.show()


if __name__=="__main__":
    main()

