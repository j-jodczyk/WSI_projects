import time
import numpy as np
import numdifftools as ndt
from matplotlib import pyplot as plt
from random import randint
import pandas as pd
from algorithms import gradientDescend, newton, newtonBtr, f
from plotting import iterToMin


def main():

    alpha = [1, 10, 100]
    n = [10, 20]
    x0 = np.array([randint(-100, 100) for i in range(10)])
    x1 = np.array([randint(-100, 100) for i in range(20)])

    steps = {
        'gradient' : [0.25, 0.09, 0.009],
        'newton' : [1, 1, 1],
        'backtracking' : [1, 1, 1]
    }

    times_of_processing = {
        'grandient' : [],
        'newton': [],
        'backtracking' : []
    }

    x2 = [-55, -64,  48, -48,  42, -68,  57, -19,  -5, -26]
    # 2 eksperymenty czasowe
    #TODO cos tu nie działa, backtracking powinien być szybszyyyyy
    # for i in range(3):
    #     t_start = time.process_time()
    #     min, next_x, num_of_iter = gradientDescend(x0, f, steps["gradient"][i], alpha[i], n[0])
    #     t_stop = time.process_time()
    #     times_of_processing["grandient"].append((t_stop-t_start, num_of_iter))

    #     t_start = time.process_time()
    #     min, next_x, num_of_iter = newton(x0, f, steps["newton"][i], alpha[i], n[0])
    #     t_stop = time.process_time()
    #     times_of_processing["newton"].append((t_stop-t_start, num_of_iter))

    #     t_start = time.process_time()
    #     min, next_x, num_of_iter = newtonBtr(x0, f, steps["backtracking"][i], alpha[i], n[0])
    #     t_stop = time.process_time()
    #     times_of_processing["backtracking"].append((t_stop-t_start, num_of_iter))

    t_start = time.process_time()
    min, next_x, num_of_iter = newton(x0, f, 0.8, 10, n[0])
    t_stop = time.process_time()
    print(t_stop-t_start)
    print(num_of_iter)
    print(min)

    t_start = time.process_time()
    min, next_x, num_of_iter = newtonBtr(x0, f, 0.8, 10, n[0])
    t_stop = time.process_time()
    print(t_stop-t_start)
    print(num_of_iter)
    print(min)


    #print(pd.DataFrame(times_of_processing))


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

    # fig4, axs = plt.subplots(2,3, sharex=True, sharey=True)
    # iterToMin(gradientDescend, fig4, axs, alpha, [[0.25, 0.15, 0.9], [0.05, 0.1, 0.09], [0.01, 0.009, 0.005]], 10, x0)
    # iterToMin(gradientDescend, fig4, axs, alpha, [[0.25, 0.15, 0.9], [0.05, 0.1, 0.09], [0.01, 0.009, 0.005]], 20, x1)

    # fig5, axs = plt.subplots(2,3, sharex=True, sharey=True)
    # iterToMin(newton, fig5, axs, alpha, [[1, 2, 0.8], [1, 2, 0.8], [1, 2, 0.8]], 10, x0)
    # iterToMin(newton, fig5, axs, alpha, [[1, 2, 0.8], [1, 2, 0.8], [1, 2, 0.8]], 20, x1)

    #fig6, axs = plt.subplots(2,3, sharex=True, sharey=True)
    #iterToMin(newtonBtr, fig6, axs, alpha, [[1, 2, 0.8], [1, 2, 0.8], [1, 2, 0.8]], 10, x0)
    #iterToMin(newtonBtr, fig6, axs, alpha, [[1, 2, 0.8], [1, 2, 0.8], [1, 2, 0.8]], 20, x1)
    #plt.show()

    #plt.savefig("iteration_to_step")


if __name__=="__main__":
    main()

