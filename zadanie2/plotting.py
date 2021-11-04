from matplotlib import pyplot as plt
from random import randint
import numpy as np
import scipy.interpolate as interpol
from algorithm import evolution_strategy, f, q


def iterToSigma(algorithm, x0_arr, func, l, s, flag, b=1):
    # czy sigma moze miec ujemne wartosci?
    # czy mamy sigme strzelic czy to jakos wyliczac
    next_y, iterations = [], []
    for i in range(len(x0_arr)):
        y, iter = algorithm(x0_arr[i], func, l, s, flag, b)[1:]
        next_y.append(y)
        iterations.append(iter)

    max_idx = iterations.index(max(iterations))
    for i in range(len(x0_arr)):
        if i!=max_idx:
            interp = interpol.interp1d(np.arange(len(next_y[i])),next_y[i])
            next_y[i] = interp(np.linspace(0,len(next_y[i])-1,len(next_y[max_idx])))
    avr = [sum(y_idx)/len(x0_arr) for y_idx in zip(*next_y)]
    return avr


x0_arr = [np.array([randint(-100, 100) for i in range(10)]) for j in range(5)]

a = np.array([1, 2, 3])
a +=a
print(a)

next_y = iterToSigma(evolution_strategy, x0_arr, f, 10, 100, True)
plt.plot(range(len(next_y)), next_y)
plt.show()
plt.savefig("result.pdf")