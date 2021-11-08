from matplotlib import pyplot as plt
from random import randint, seed
import numpy as np
import scipy.interpolate as interpol
from algorithm import evolution_strategy, f, q


def plot(seeds, fig, axs, algorithm, x0, func, lambda_, sigma, ni=None,
                       max_func_budget=None, b=1, epsilon=0.0000001, max_diff_cout=20): # s=0.1, 1, 10, 0.01

    mut_type = ['SA', 'LMR']
    fig.text(0.5, 0.04, 'iteration', ha='center')
    fig.text(0.04, 0.5, 'f(x)', va='center', rotation='vertical')
    for k in range(len(lambda_)):
        for j in range(len(sigma)):
            for m in mut_type:
                f_min_arr, next_y, iterations = [], [], []
                for i in range(5):
                    np.random.seed(seeds[i])
                    f_min, y, iter = algorithm(x0, func, lambda_[k], sigma[j], m, ni, max_func_budget, b, epsilon, max_diff_cout)[1:]

                    next_y.append(y)
                    iterations.append(iter)
                    f_min_arr.append(f_min)
                max_lenght = max(iterations)
                for i in range(5):
                    if len(next_y[i])<max_lenght:
                        next_y[i] += [next_y[i][-1]]*(max_lenght - len(next_y[i]))
                avr = np.average(next_y, axis=0)
                print(f'lambda={lambda_[k]}, sigma={sigma[j]}, mut={m}, f_min={sum(f_min_arr)/5}')
                axs[k][j].plot(range(max_lenght), avr, label=f'{m}')

            axs[k][j].set_yscale('log')
            axs[k][j].title.set_text(f"lambda={lambda_[k]}, sigma={sigma[j]}")
            axs[k][j].title.set_size(7)
            axs[k][j].legend()
