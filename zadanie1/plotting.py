import numpy as np
import numdifftools as ndt
from matplotlib import pyplot as plt

from algorithms import gradientDescend, newton, newtonBtr, inv_hess, f


def iterToMin(algorithm, fig, axs, alpha, step, n, x0):
    k = 0 if n==10 else 1
    color = ['g', 'b', 'r']
    fig.text(0.5, 0.04, 'iteration', ha='center')
    fig.text(0.04, 0.5, 'f(x)', va='center', rotation='vertical')
    for j in range(len(alpha)):
        for i in range(len(step[j])):
            f_min, next_y, num_of_iter = algorithm(x0, f, step[j][i], alpha[j], n)
            axs[k][j].plot(range(num_of_iter+1), next_y, f'{color[i]}', label=f'step={step[j][i]}')
        axs[k][j].set_yscale('log')
        axs[k][j].title.set_text(f"alpha={alpha[j]}, n={n}")
        axs[k][j].title.set_size(7)
        axs[k][j].legend()