import numpy as np
from math import e

#matplotlib.contour
#punkt poczatkowy = centroid
#mozna wykres o tym jak sie zmienia sigma w kolejnych iteracjach
# premend zamiast append - bo append ma zlozonosc o(n) [musi przejsc przez cala liste], a premend o(1)


def f(x):
    """
    spherical function.
    """
    result = 0
    for i in range(len(x)):
        result += x[i]**2
    return result


def q(x):
    """
    other tested function.
    """
    D = len(x)
    norm_pow_2 = np.linalg.norm(x)**2
    return((norm_pow_2 - D)**2)**(1/8)+(1/D)*(1/2*norm_pow_2+sum(x))+1/2


def evolution_strategy(x0, func, lambda_, sigma, mut_type, ni=None,
                       max_func_budget=None, b=1, epsilon=0.0000001, max_diff_cout=20): # nie flaga LMR, tylko mapa na LMR i SA
    """
    Implements (n/n,lambda) ES with self-adaptation.

    : param x0: array of starting co-ordinates
    : param func: objective function (plays role of fitness function)
    : param lambda_: how many offsprings will algorithm generate
    : param sigma: mutation parameter
    : param mut_type: takes values 'SA' of 'LMR', decides of what mutation will be used
    : param ni: how many best offsprings will be chosen from lambda created
    : max_func_budget: how many calls of objective function are to be performed before stopping algorithm
    : param b: tau = b*1/(D**(1/2)) where D is dimentionality of x
    : param epsilon: sets error margin for difference between next two x values
    : param max_diff_cout: how many times difference goes below epsilon before stopping the algorithm

    returns: ninumum x, func(minumum x), list of next values of func, number of iterations
    """
    #TODO czy moze byc taki sposob okreslania mutacji + czy zabezpieczenie przed zla nazwa potrzebne
    D = len(x0)
    tau = b*1/(D**(1/2))
    ni = ni if ni else int(lambda_/2)
    max_func_budget = max_func_budget if max_func_budget else 1000*D # moze byc taki? bo inaczej nie znajduje az tak szybko
    func_budget = 0
    iterations = 0
    surv_x = x0
    next_y = []

    diff_cout = 0

    while func_budget<=max_func_budget and diff_cout<=max_diff_cout: #TODO czy moze byc taki warunek?

        prev_x = surv_x

        offsprings = np.empty((0, D))
        offspr_sigma = np.array([])
        y = []

        for i in range(lambda_):  # generate lambda_ offsprings of surv_x
            xi = tau*np.random.normal(0, 1)
            z = np.random.multivariate_normal(np.zeros(D), np.eye(D))
            # mutation:
            if mut_type=="SA":
                offspr_sigma = np.append(offspr_sigma, sigma*np.exp(xi))
                offsprings = np.append(offsprings, [surv_x + offspr_sigma[i]*z], axis=0)
            else:
                offsprings = np.append(offsprings, [surv_x + sigma*z], axis=0)
            y.append(func(offsprings[i]))  # evaluation of offsprings
            func_budget += 1
        idx = np.argpartition(y, ni)  # devide offsprings based on the smallest value of func
        offsprings = offsprings[idx[:ni]]  # only ni offsprings with smallest value of func survive
        # calculate new value of mutation parameter
        if mut_type=="SA":
            offspr_sigma = offspr_sigma[idx[:ni]]
            sigma = sum([offspr_sigma[i]*1/ni for i in range(ni)])
        else:
            sigma *= e**(tau*np.random.normal(0, 1))
        surv_x = sum([offsprings[i]*1/ni for i in range(ni)])  # calculate new surviving x
        iterations+=1
        next_y.append(func(surv_x))

        #TODO tu powinno byc sprawdzanie roznicy miedzy x czy func(x)?
        if np.linalg.norm(abs(surv_x-prev_x))<epsilon:  # check whether next x are close to each other
            diff_cout += 1

    return surv_x, func(surv_x), next_y, iterations


