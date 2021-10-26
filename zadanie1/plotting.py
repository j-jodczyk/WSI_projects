import numpy as np
import numdifftools as ndt
from matplotlib import pyplot as plt
from functools import partial

from algorithms import gradientDescend, newton, newtonBtr, f


def iterToMin(algorithm, fig, axs, a, step, n0, x0):
    k = 0 if n0==10 else 1
    color = ['g', 'b', 'r']
    fig.text(0.5, 0.04, 'iteration', ha='center')
    fig.text(0.04, 0.5, 'f(x)', va='center', rotation='vertical')
    z = 1
    for j in range(len(a)):
        for i in range(len(step[j])):
            func = partial(f, alpha=a[j], n=n0)
            message, x, f_min, iter, next_y  = algorithm(x0, func, step[j][i])
            axs[k][j].plot(range(iter+1), next_y, f'{color[i]}', label=f'step={step[j][i]}')
            print(f"{z} done")
            z +=1
        axs[k][j].set_yscale('log')
        axs[k][j].title.set_text(f"alpha={a[j]}, n={n0}")
        axs[k][j].title.set_size(7)
        axs[k][j].legend()