# Zadanie 1. (7p)
# Zaimplementuj algorytm gradientu prostego oraz algorytm Newtona.
# Algorytm Newtona powinien móc działać w dwóch trybach:
#       ze stałym parametrem kroku
#       z adaptacją parametru kroku przy użyciu metody z nawrotami.
#           w iteracyjny sposób znaleźć

import numpy as np
import time
import numdifftools as ndt
from matplotlib import pyplot as plt
from random import randint

# TODO dziedzina ma być jakoś sprawdzana?
def gradientDescend(x0, func, step, alpha, n, error_margin=0.0001, max_num_of_iter=1000):
    x = x0
    diff = x0
    num_of_iterations = 0
    y = [func(x0, alpha, n)]
    # TODO czy takie kryterium stopu moze byc, czy lepiej z gradientem
    # stop criterion:
    while np.linalg.norm(diff)>error_margin and num_of_iterations<max_num_of_iter:
        prev_x = x
        d = ndt.Gradient(func)(prev_x, alpha, n)
        x = prev_x - step*d

        # TODO: czy takie zmniejszanie zostawić, czy polegać na dobrym doborze kroku?
        # if step makes the function grow - make the step smaller
        while func(x, alpha, n) > func(prev_x, alpha, n):
            step *= 0.95
            x = prev_x - step*d

        diff = prev_x-x

        num_of_iterations += 1
        y.append(func(x, alpha, n))

    return (x, y, num_of_iterations)


def newton(x0, func, step, alpha, n, error_margin=0.0001, max_num_of_iter=1000):
    x = x0
    diff = x0
    num_of_iterations = 0
    y = [func(x0, alpha, n)]
    hessian = ndt.Hessian(f)
    # estymacja hessjanu:
    prev_inv_hess = np.linalg.inv(np.eye(n))

    while np.linalg.norm(diff)>error_margin  and num_of_iterations<max_num_of_iter:
        prev_x = x
        # d : hess(x)**(-1)*grad(x)
        #d = np.linalg.inv(hessian(prev_x, alpha, n))@ndt.Gradient(func)(prev_x, alpha, n)
        # estymacja hesjanu:
        d = np.asarray(prev_inv_hess@ndt.Gradient(func)(prev_x, alpha, n)).reshape(-1)

        x = prev_x - step*d

        #TODO: czy tu też należy sprawdzać czy f(x)>f(prev_x) czy tym zajmuje się hessian?
        diff = prev_x-x

        num_of_iterations += 1
        y.append(func(x, alpha, n))

        # dla estymacji hesjanu
        x_diff = np.transpose(np.matrix(np.subtract(prev_x, x)))
        grad_diff = np.transpose(np.matrix(np.subtract(ndt.Gradient(func)(prev_x, alpha, n), ndt.Gradient(func)(x, alpha, n))))
        prev_inv_hess = inv_hess(prev_inv_hess, x_diff, grad_diff)

    return (x, y, num_of_iterations)


def newtonBacktracking(x0, func, step, alpha, n, betha=0.95, gamma=0.01, error_margin=0.0001, max_num_of_iter=1000):
    x = x0
    diff = x0
    prev_x = x
    num_of_iterations = 0
    y = [func(x0, alpha, n)]
    hessian = ndt.Hessian(f)
    # estymacja hessjanu:
    # prev_inv_hess = np.linalg.inv(np.eye(n))
    #v = -prev_inv_hess@ndt.Gradient(func)(prev_x, alpha, n)
    v = -np.linalg.inv(hessian(prev_x, alpha, n))@ndt.Gradient(func)(prev_x, alpha, n)

    while np.linalg.norm(diff)>error_margin and num_of_iterations<max_num_of_iter:
        prev_x = x

        # d = hess(x)**(-1)*grad(x)
        d = np.linalg.inv(hessian(prev_x, alpha, n))@ndt.Gradient(func)(prev_x, alpha, n)
        # esytmacja hesjanu
        #d = np.asarray(prev_inv_hess@ndt.Gradient(func)(prev_x, alpha, n)).reshape(-1)

        # minimalizacja t
        t = 1

        a = func(x, alpha, n)+gamma*t*np.transpose(ndt.Gradient(func)(x, alpha, n))@v
        b = func(x, alpha, n)
        # while func(x+t*v, alpha, n)>func(x, alpha, n)+gamma*t*np.transpose(ndt.Gradient(func)(x, alpha, n))@v:
        #     t = betha*t

        while func(x+t*v, alpha, n)>func(x, alpha, n)+gamma*t*np.transpose(ndt.Gradient(func)(x, alpha, n))@(-d):
            t = betha*t

        x = prev_x - t*step*d
        diff = prev_x-x

        num_of_iterations += 1
        y.append(func(x, alpha, n))

        # dla estymacji hesjanu
        # x_diff = np.transpose(np.matrix(np.subtract(prev_x, x)))
        # grad_diff = np.transpose(np.matrix(np.subtract(ndt.Gradient(func)(prev_x, alpha, n), ndt.Gradient(func)(x, alpha, n))))
        # prev_inv_hess = inv_hess(prev_inv_hess, x_diff, grad_diff)

    return (x, y, num_of_iterations)

# given function
def f(x, alpha, n):
    result = 0
    for i in range(n):
        result += alpha**(i/(n-1))*x[i]**2
    return result

# DFP inverted hessian estimation
def inv_hess(prev_inv_hess, x_diff, grad_diff):
    x_diff_t = np.transpose(x_diff)
    grad_diff_t = np.transpose(grad_diff)

    return prev_inv_hess+x_diff@x_diff_t/(x_diff_t@grad_diff)-(prev_inv_hess@grad_diff@grad_diff_t@prev_inv_hess)/(grad_diff_t@prev_inv_hess@grad_diff)


# plotting function no.1
def iterToStep(algorithm, fig, alpha, step, n, x0):
    min_step = []
    k = 0 if n==10 else 3
    for j in range(len(alpha)):
        iterations = []
        min_y = []
        for i in range(1, len(step-2)):
            f_min, next_y, num_of_iter = algorithm(x0, f, step[i], alpha[j], n)
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


#plotting function no.2
def iterToMin(algorithm, fig, alpha, step, n, x0):
    k = 0 if n==10 else 3
    color = ['g', 'b', 'r']
    for j in range(len(alpha)):
        ax = fig.add_subplot(2, 3, j+k+1)
        for i in range(len(step[j])):
            f_min, next_y, num_of_iter = algorithm(x0, f, step[j][i], alpha[j], n)
            plt.plot(range(num_of_iter+1), next_y, f'{color[i]}', label=f'step={step[j][i]}')
        ax.title.set_text(f"alpha={alpha[j]}, n={n}")
        ax.title.set_size(7)
        ax.legend()
        plt.xlabel("iteration", fontsize=6)
        plt.ylabel("f(x)", fontsize=6)



def main():

    alpha = [1, 10, 100]
    n = [10, 20]
    #step = np.linspace(0, 0.1, 50) #potrzebe nowe wykresy
    x0 = np.array([randint(1, 100) for i in range(10)])
    x1 = np.array([randint(1, 100) for i in range(20)])

    # time
    # t_start = time.process_time()
    # min, next_x, next_y, num_of_iter = localMinimum('gradientDescend', x0, f, 0.3, alpha[0], n[0])
    # t_stop = time.process_time()
    # print(f"gradient process time:{t_stop-t_start}")
    # print(num_of_iter)
    # print(min)

    t_start = time.process_time()
    min, next_x, num_of_iter = newton(x0, f, 0.3, alpha[0], n[0])
    t_stop = time.process_time()
    print(f"newton process time:{t_stop-t_start}")
    print(num_of_iter)
    print(min)

    # t_start = time.process_time()
    # min, next_x, next_y, num_of_iter = localMinimum('newtonBacktracking', x0, f, 0.9, alpha[0], n[0])
    # t_stop = time.process_time()
    # print(f"newton with backtracking process time:{t_stop-t_start}")
    # print(num_of_iter)
    # print(min)

    #plotting
    # fig = plt.figure()
    # min_step = iterToStep(gradientDescend, fig, alpha, step, 10, x0)
    # min_step += iterToStep(gradientDescend, fig, alpha, step, 20, x1)
    # print(min_step)

    # fig2 = plt.figure()
    # min_step = iterToStep(newton, fig2, alpha, step, 10, x0)
    # min_step += iterToStep(newton, fig2, alpha, step, 20, x1)
    # print(min_step)

    # fig3 = plt.figure()
    # min_step = iterToStep(newtonBacktracking, fig3, alpha, step, 10, x0)
    # min_step += iterToStep(newtonBacktracking, fig3, alpha, step, 20, x1)
    # print(min_step)

    # fig4 = plt.figure()
    # iterToMin(gradientDescend, fig4, alpha, [[0.25, 0.4, 0.9], [0.05, 0.02, 0.005], [0.005, 0.001, 0.0001]], 10, x0)
    # iterToMin(gradientDescend, fig4, alpha, [[0.25, 0.4, 0.9], [0.05, 0.02, 0.005], [0.005, 0.001, 0.0001]], 20, x1)

    # fig5 = plt.figure()
    # iterToMin(newton, fig5, alpha, [[0.3, 0.4, 0.8], [0.1, 0.02, 0.5], [0.5, 0.9, 0.2]], 10, x0)
    # iterToMin(newton, fig5, alpha, [[0.3, 0.4, 0.8], [0.1, 0.02, 0.5], [0.5, 0.9, 0.2]], 20, x1)

    #fig6 = plt.figure()
    #iterToMin(newtonBacktracking, fig6, [alpha[0]], [[0.9, 10, 5]], 10, x0)
    # iterToMin(newtonBacktracking, fig6, alpha, [[0.25, 0.4, 0.9], [0.001, 0.02, 0.005], [0.005, 0.001, 0.0001]], 20, x1)
    plt.show()

    plt.savefig("iteration_to_step")


if __name__=="__main__":
    main()

