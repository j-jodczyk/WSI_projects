import numpy as np


def f(x, alpha, n):
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
    return((norm_pow_2 - D)**2)**(1/8)+(1/D)*(1/2*norm_pow_2**2+sum(x))+1/2


def evolution_strategy(x0, func, l, s, b=1):
    """
    Implements (n/n,lambda) ES

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
    xi, z, x, offspr_s = []
    while True:
        for i in range(l):
            xi[i] = tau*np.random.normal(0, 1)
            #TODO czy dobrze rozumiem ten rozklad normalny:
            z[i] = np.random.multivariate_normal(np.zeros(D), np.eye(D))
            offspr_s[i] = s*np.exp(xi[i])
            x[i] = surv_x + s[i]*z[i]
        # somewhere here we choose n offsprings to keep
        s = sum([offspr_s[i]*1/n for i in range(n)])
        surv_x = sum([x[i]*1/n for i in range(n)])
