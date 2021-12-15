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
import cvxopt as cvx


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
        self.kernel = np.zeros((len(X), len(X)))
        self.calculate_kernel()

    def calculate_kernel(self):
        N = len(self.X)
        for i in range(N):
            for j in range(N):
                self.kernel[i][j] = self.kernel_function(self.X[i], self.X[j])

    def train(self):
        N = len(self.X)

        A = self.Y.reshape(1, N)
        A = A.astype('float')

        P = cvx.matrix((np.outer(self.Y, self.Y) * self.kernel))
        q = cvx.matrix(-np.ones((N,1)))
        G = cvx.matrix(np.vstack((np.eye(N) * -1, np.eye(N))))
        h = cvx.matrix(np.hstack((np.zeros(N), np.ones(N) * self.C)))
        A = cvx.matrix(A)
        b = cvx.matrix(0.0)

        cvx.solvers.options['show_progress'] = False
        solution = cvx.solvers.qp(P, q, G, h, A, b)
        alpha = np.array(solution["x"])
        return alpha




def main(filename, kernel, C, gamma=1):
    df = pd.read_csv(filename, delimiter=';', low_memory=False)
    discretization(df)
    X = df.drop(columns=['quality']).copy()
    Y = df['quality']

    # split data into train set 0.8, test set 0.1 and validation set 0.1
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, train_size=0.8)

    if kernel == 'gausian_kernel':
        kernel_function = partial(gausian_kernel, gamma)
        mess = f'C: {C}, gamma: {gamma}, '
    else:
        kernel_function = linear_kernel
        mess = f'C: {C}, '

    svm = SVM(X_train.to_numpy(), Y_train.to_numpy(), C, kernel_function)
    alpha = svm.train()

    Y_pred = []
    for x in X_test.to_numpy():
        Y_pred.append(np.sign(np.sum([alpha[i]*Y_train.to_numpy()[i]*kernel_function(X_train.to_numpy()[i], x) for i in range(len(alpha))])))

    print( mess+f'acc: {accuracy_score(Y_test, Y_pred)}\n')


if __name__=="__main__":
    filename = f'/home/julia/PAP/pap/zadanie4/{sys.argv[1]}'
    kernel = sys.argv[2]
    C = sys.argv[3]
    gamma = sys.argv[4] if sys.argv[4] else 1
    main(filename, kernel, C, gamma)