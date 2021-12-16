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
    def __init__(self, X_train, Y_train, X_test, Y_test, C, kernel_function, min_alpha=None):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.C = C
        self.min_alpha = min_alpha if min_alpha else 1e-5
        self.kernel_function = kernel_function
        self.alpha = np.zeros(len(X_train))
        self.b = 0
        self.kernel = np.zeros((len(X_train), len(X_train)), dtype=float)
        self.calculate_kernel()

    def calculate_kernel(self):
        N = len(self.X_train)
        for i in range(N):
            for j in range(N):
                self.kernel[i][j] = self.kernel_function(self.X_train[i], self.X_train[j])

    def train(self):

        n_samples, n_features = self.X_train.shape
        P = cvxopt.matrix(np.outer(self.Y_train,self.Y_train) * self.kernel, tc='d') #type code dubble
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(self.Y_train, (1, n_samples), tc='d')
        b = cvxopt.matrix(0.0)
        G = cvxopt.matrix(np.vstack((np.eye(n_samples) * -1, np.eye(n_samples))))
        h = cvxopt.matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C)))

        cvxopt.solvers.options['show_progress'] = False
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        alpha = np.ravel(solution['x'])

        # support vectors have non zero alpha values
        idx = alpha > self.min_alpha
        self.alpha = alpha[idx]
        self.new_x = self.X_train[idx]
        self.new_y = self.Y_train[idx]

        self.b = self.new_y[0]
        for i in range(len(self.alpha)):
            self.b -= self.alpha[i] * self.new_y[i] * self.kernel[i][0]

        return (self.alpha, self.b)

    def predict(self):
        self.alpha, self.b = self.train()
        Y_pred = []
        for x in self.X_test:
            temp = np.sum([self.alpha[i]*self.new_y[i]*self.kernel_function(self.new_x[i], x)
            for i in range(len(self.alpha))])
            Y_pred.append(np.sign(temp + self.b))

        return accuracy_score(self.Y_test, Y_pred)



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

    # split data into train set 0.8, test set 0.2
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, train_size=0.8, random_state=10)

    if kernel == 'gausian_kernel':
        kernel_function = partial(gausian_kernel, gamma=gamma)
    else:
        kernel_function = linear_kernel

    svm = SVM(X_train, Y_train.to_numpy(), X_test, Y_test.to_numpy(), C, kernel_function)

    accuracy_score = svm.predict()
    if kernel == 'gausian_kernel':
        print(C, gamma, accuracy_score)
    else:
        print(C, accuracy_score)


if __name__=="__main__":
    main(sys.argv)
    # argv[1] = filename
    # argv[2] = kernel functon: linear or gausian_kernal
    # argv[3] = C
    # argv[4] = gamma