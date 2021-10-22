import time
import numpy as np
import numdifftools as ndt
from matplotlib import pyplot as plt
from random import randint

from algorithms import gradientDescend, newton, newtonBacktracking, inv_hess, f


def main():

    alpha = [1, 10, 100]
    n = [10, 20]
    #step = np.linspace(0, 0.1, 50) #potrzebe nowe wykresy
    x0 = np.array([randint(1, 100) for i in range(10)])
    x1 = np.array([randint(1, 100) for i in range(20)])

    time
    t_start = time.process_time()
    min, next_x, num_of_iter = gradientDescend(x0, f, 0.9, alpha[0], n[0])
    t_stop = time.process_time()
    print(f"gradient process time:{t_stop-t_start}")
    print(num_of_iter)
    print(min)

    t_start = time.process_time()
    min, next_x, num_of_iter = newton(x0, f, 0.9, alpha[0], n[0])
    t_stop = time.process_time()
    print(f"newton process time:{t_stop-t_start}")
    print(num_of_iter)
    print(min)

    t_start = time.process_time()
    min, next_x, num_of_iter = newtonBacktracking(x0, f, 0.9, alpha[0], n[0])
    t_stop = time.process_time()
    print(f"newton with backtracking process time:{t_stop-t_start}")
    print(num_of_iter)
    print(min)

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
    #plt.show()

    #plt.savefig("iteration_to_step")


if __name__=="__main__":
    main()

