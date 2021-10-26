import time
import numpy as np
import numdifftools as ndt
from matplotlib import pyplot as plt
from random import randint
import pandas as pd
from functools import partial
from algorithms import gradientDescend, newton, newtonBtr, f
from plotting import iterToMin



def main():

    a = [1, 10, 100]
    n0 = [10, 20]
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

    print(x0)
    print(x1)

    # for i in range(3):
    #     t_start = time.process_time()
    #     message, x, f_min, iter, y = gradientDescend(x0, partial(f, alpha=a[i], n=n0[0]), steps["gradient"][i])
    #     t_stop = time.process_time()
    #     times_of_processing["grandient"].append((t_stop-t_start, iter))

    #     t_start = time.process_time()
    #     message, x, f_min, iter, y = newton(x0, partial(f, alpha=a[i], n=n0[0]), steps["newton"][i])
    #     t_stop = time.process_time()
    #     times_of_processing["newton"].append((t_stop-t_start, iter))

    #     t_start = time.process_time()
    #     message, x, f_min, iter, y = newtonBtr(x0, partial(f, alpha=a[i], n=n0[0]), steps["backtracking"][i])
    #     t_stop = time.process_time()
    #     times_of_processing["backtracking"].append((t_stop-t_start, iter))


    # print(pd.DataFrame(times_of_processing))

    x0 = [ 84, -27, 8, 74, 59, -81, 14, -85, 78, -65]
    # x1 = [ 33, 17, 10, -69, 79, 78, -85, 7, 85, -47, -15, 74, -71, -41, 22, -85, 23, -76, 5, -42]
    #plotting - testy muszą być na tej samej wartości :/ - seed ustawić
    # fig4, axs = plt.subplots(2,3, sharex=True, sharey=True)
    # iterToMin(gradientDescend, fig4, axs, a, [[0.25, 0.15, 0.9], [0.05, 0.1, 0.09], [0.01, 0.009, 0.005]], 10, x0)
    # iterToMin(gradientDescend, fig4, axs, a, [[0.25, 0.15, 0.9], [0.05, 0.1, 0.09], [0.01, 0.009, 0.005]], 20, x1)

    # fig5, axs = plt.subplots(2,3, sharex=True, sharey=True)
    # iterToMin(newton, fig5, axs, a, [[1, 0.4, 0.8], [0.1, 1, 0.5], [0.1, 1, 0.5]], 10, x0)
    func = partial(f, alpha=1, n=10)
    message, x, f_min, iter, next_y = newton(x0, func, 1.1)
    print(message)
    print(iter)
    message, x, f_min, iter, next_y = newtonBtr(x0, func, 1.1)
    print(message)
    print(iter)
    # iterToMin(newton, fig5, axs, a, [[1, 0.4, 0.8], [0.1, 1, 0.5], [0.1, 1, 0.5]], 20, x1)

    # fig6, axs = plt.subplots(2,3, sharex=True, sharey=True)
    # iterToMin(newtonBtr, fig6, axs, a, [[1, 0.4, 0.8], [0.1, 1, 0.5], [0.1, 1, 0.5]], 10, x0)
    # iterToMin(newtonBtr, fig6, axs, a, [[1, 0.4, 0.8], [0.1, 1, 0.5], [0.1, 1, 0.5]], 20, x1)
    plt.show()

    #plt.savefig("iteration_to_step")


if __name__=="__main__":
    main()

