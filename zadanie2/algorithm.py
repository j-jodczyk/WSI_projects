import numpy as np
from math import e
from random import randint

#TODO czy mamy porówywać czasowo te algorytmy?

def f(x, alpha=1, n=10):
    #TODO czy tu już alpha = 1, n = 10?
    """
    spherical function.
    """
    result = 0
    for i in range(n):
        result += alpha**(i/(n-1))*x[i]**(2)
    return result


def q(x):
    """
    other tested function.
    """
    # TODO D = len(x)?
    D = len(x)
    norm_pow_2 = np.linalg.norm(x)**2
    return((norm_pow_2 - D)**2)**(1/8)+(1/D)*(1/2*norm_pow_2+sum(x))+1/2


def evolution_strategy_SA(x0, func, l, s, b=1, max_iter=5000, epsilon=0.000001):
    """
    Implements (n/n,lambda) ES with self-adaptation.

    : param x0: array of starting co-ordinates
    : param func: objective function
    : param l: lambda
    : param s: sigma
    : param b: tau = b*1/(D**(1/2)) where D is dimentionality of x

    returns: #TODO what returns
    """
    D = len(x0)
    n = int(l/2)
    tau = b*1/(D**(1/2))
    iteration = 0
    surv_x = x0
    func_budget = 0
    iterations = 0
    diff = surv_x = x0
    while not (func_budget>100*D or iterations>max_iter or np.linalg.norm(diff)<epsilon):
        func_budget=0
        xi = [] #TODO czy w sumie potrzebujemy xi i z zapisywac jako liste?
        z = []
        x = np.empty((0, D))
        offspr_s = np.array([])
        y = []
        prev_x = surv_x
        for i in range(l):
            xi.append(tau*np.random.normal(0, 1))
            #TODO czy dobrze rozumiem ten rozklad normalny:
            z.append(np.random.multivariate_normal(np.zeros(D), np.eye(D)))
            offspr_s = np.append(offspr_s, s*np.exp(xi[i]))
            x = np.append(x, [surv_x + offspr_s[i]*z[i]], axis=0)
            y.append(func(x[i]))
            func_budget += 1
        idx = np.argpartition(y, n)
        offspr_s = offspr_s[idx[:n]]
        x = x[idx[:n]]
        s = sum([offspr_s[i]*1/n for i in range(n)])
        surv_x = sum([x[i]*1/n for i in range(n)])
        diff = abs(surv_x-prev_x)
        iterations+=1
    return func(surv_x)


def evolution_strategy_LMR(x0, func, l, s, b=1, max_iter=5000, epsilon=0.000001):
    """
    Implements (n/n,lambda) ES with Log-Normal Mutation Rule.

    parameters and return same as evolution_stategy_SA
    """
    D = len(x0)
    n = int(l/2)
    tau = b*1/(D**(1/2))
    iteration = 0
    surv_x = x0
    func_budget = 0
    iterations = 0
    diff = surv_x = x0
    while not (func_budget>100*D or iterations>max_iter or np.linalg.norm(diff)<epsilon):
        func_budget=0
        z = []
        x = np.empty((0, D))
        y = []
        prev_x = surv_x
        for i in range(l):
            z.append(np.random.multivariate_normal(np.zeros(D), np.eye(D)))
            x = np.append(x, [surv_x + s*z[i]], axis=0)
            y.append(func(x[i]))
            func_budget += 1
        idx = np.argpartition(y, n)
        x = x[idx[:n]]
        s *= e**(tau*np.random.normal(0, 1))
        surv_x = sum([x[i]*1/n for i in range(n)])
        diff = abs(surv_x-prev_x)
        iterations+=1
    return func(surv_x)


x0 = np.array([randint(-100, 100) for i in range(10)])
print(evolution_strategy_LMR(x0, q, 10, 8))
