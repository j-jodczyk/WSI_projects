from matplotlib import pyplot as plt
from random import randint
import numpy as np
import scipy.interpolate as interpol
from algorithm import evolution_strategy, f, q


def iterToSigma(algorithm, x0_arr, func, lambda_, sigma, mut_type, ni=None,
                       max_func_budget=None, b=1, epsilon=0.0000001, max_diff_cout=20): # s=0.1, 1, 10, 0.01
    next_y, iterations = [], []
    for i in range(len(x0_arr)):
        y, iter = algorithm(x0_arr[i], func, lambda_, sigma, mut_type, ni, max_func_budget, b, epsilon, max_diff_cout)[2:]
        next_y.append(y)
        iterations.append(iter)
        print(y[-1])

    max_lenght = max(iterations)
    for i in range(0, len(x0_arr)):
        if len(next_y[i])<max_lenght:
            next_y[i] += [next_y[i][-1]]*(max_lenght - len(next_y[i]))
    avr = np.average(next_y, axis=0)
    return avr, max_lenght


x0_arr = [np.random.random(10)*100 for j in range(5)]

next_y, iter = iterToSigma(evolution_strategy, x0_arr, f, 10, 1, 'SA')
plt.plot(range(iter), next_y)
plt.show()
plt.savefig("result.pdf")