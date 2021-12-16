#zdyskretyzuj zmienna objaśniana - ocena wina od 1 do 10 ma przyjmować wartości 0(np mniejsze od 5)/1(np większe równe 5)
#funkcja jadrowa
#hiperparametry c - jak powaznie svm tratuje odchylenia, gamma - uzywana przy rbf
#implementacja logiki algorytmu, ale nie musimy pisac algorytmow optymalizacyjnych sami
#utworzenie zbioru testujacego i uczacego tez mozna z gotowego rozwiazania
#nacisk na sprawozdanie
#metryki jak dobrze dziala algorytm - accuracy
#zobaczyc jak sie rozklada ocena win - zeby zbior byl zrownowazony

import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import pandas as pd
import numpy as np
from functools import partial
import cvxopt



def discretization(dataframe):
    dataframe['quality'].values[dataframe['quality'] <= 5] = -1
    dataframe['quality'].values[dataframe['quality'] > 5] = 1

def gausian_kernel(u, v, gamma):
    return np.exp(gamma * (-np.linalg.norm(u-v)**2))

def linear_kernel(u, v):
    return np.dot(u, v)


class SVM:
    def __init__(self, X, Y, C, kernel_function):
        self.X = X
        self.Y = Y
        self.C = C
        self.kernel_function = kernel_function
        self.alpha = np.zeros(len(X))
        self.kernel = np.zeros((len(X), len(X)), dtype=float)
        self.calculate_kernel()

    def calculate_kernel(self):
        N = len(self.X)
        for i in range(N):
            for j in range(N):
                self.kernel[i][j] = self.kernel_function(self.X[i], self.X[j])

    def train(self):

        n_samples, n_features = self.X.shape
        P = cvxopt.matrix(np.outer(self.Y,self.Y) * self.kernel, tc='d') #type code dubble
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(self.Y, (1, n_samples), tc='d')
        b = cvxopt.matrix(0.0)
        tmp1 = np.identity(n_samples) *(-1)
        tmp2 = np.identity(n_samples)
        G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
        tmp1 = np.zeros(n_samples)
        tmp2 = np.ones(n_samples) * self.C
        h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        cvxopt.solvers.options['show_progress'] = False
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        alpha = np.array(solution["x"])

        a = np.ravel(solution['x'])

        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        a1 = a[sv]
        sv1 = self.X[sv]
        sv_y = self.Y[sv]

        # Intercept
        b = 0
        for n in range(len(a1)):
            b += sv_y[n]
            b -= np.sum(a1 * sv_y * self.kernel[ind[n],sv])
        b /= len(a)

        return (alpha, b)


def main(argv):
    filename = f'/home/julia/PAP/pap/zadanie4/{sys.argv[1]}'
    kernel = sys.argv[2]
    C = float(sys.argv[3])
    gamma = float(sys.argv[4]) if sys.argv[4] else 1.0

    df = pd.read_csv(filename, delimiter=';', low_memory=False)
    discretization(df)
    X = df.drop(columns=['quality']).copy()
    X = preprocessing.normalize(X, axis=0)
    Y = df['quality']

    # split data into train set 0.8, test set 0.1 and validation set 0.1
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, train_size=0.8)

    if kernel == 'gausian_kernel':
        kernel_function = partial(gausian_kernel, gamma=gamma)
        mess = f'C: {C}, gamma: {gamma}, '
    else:
        kernel_function = linear_kernel
        mess = f'C: {C}, '

    svm = SVM(X_train, Y_train.to_numpy(), C, kernel_function)
    alpha, b = svm.train()

    Y_pred = []
    for x in X_test:
        temp = np.sum([alpha[i]*Y_train.to_numpy()[i]*kernel_function(X_train[i], x) for i in range(len(alpha))])
        Y_pred.append(np.sign(temp + b))

    print( mess+f'acc: {accuracy_score(Y_test, Y_pred)}\n')


if __name__=="__main__":
    main(sys.argv)
    # argv[1] = filename
    # argv[2] = kernel functon: linear or gausian_kernal
    # argv[3] = C
    # argv[4] = gamma