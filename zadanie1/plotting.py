import numpy as np
import numdifftools as ndt
from matplotlib import pyplot as plt

from algorithms import gradientDescend, newton, newtonBacktracking, inv_hess, f


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