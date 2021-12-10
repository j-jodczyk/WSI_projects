#zdyskretyzuj zmienna objaśniana - ocena wina od 1 do 10 ma przyjmować wartości 0(np mniejsze od 5)/1(np większe równe 5)
#funkcja jadrowa
#hiperparametry c - jak powaznie svm tratuje odchylenia, gamma - uzywana przy rbf
#implementacja logiki algorytmu, ale nie musimy pisac algorytmow optymalizacyjnych sami
#utworzenie zbioru testujacego i uczacego tez mozna z gotowego rozwiazania
#nacisk na sprawozdanie
#metryki jak dobrze dziala algorytm - accuracy
#zobaczyc jak sie rozklada ocena win - zeby zbior byl zrownowazony

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from functools import partial
import cvxopt as cvx


def discretization(dataframe):
    dataframe['quality'].values[dataframe['quality'] <= 5] = -1
    dataframe['quality'].values[dataframe['quality'] > 5] = 1

def gausian_kernel(u, v, gamma):
        return np.exp(gamma * (-np.linalg.norm(u-v)**2))


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

    def objectiveFunction(self, x):
        result = 0
        for i in range(len(self.X)) :
            result+=self.alpha[i]*self.Y[i]*self.kernel_function(self.X[i], x)
        return result

    def kernel_trick(alpha, y, x, X, kernel_func):
        result = 0
        for i in len(X):
            result+=alpha[i]*y[i]*kernel_func(X[i], x)
        return result

    def distance(self):
        dist = np.array([])
        for i in range(len(self.X)):
            d = 1 - self.Y[i]*self.objectiveFunction(self.X[i])
            np.append(dist, d)
        dist[dist<0]=0
        return dist

    def omega(self):
        result = 0
        for i in range(len(self.X)):
            for j in range(len(self.X)):
                result+=self.alpha[i]*self.alpha[j]*self.Y[i]*self.Y[j]*self.kernel_function(self.X[i], self.X[j])
        return result

    def train(self):
        N = len(self.X)
        print("Started training")
        #what is even happening
        H = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                H[i][j] = self.Y[i]*self.Y[j]*self.kernel[i][j]
        A = self.Y.reshape(1, -1)
        A = A.astype('float')

        P = cvx.matrix(H)
        q = cvx.matrix(np.ones(N) * -1)
        G = cvx.matrix(np.negative(np.eye(N)))
        h = cvx.matrix(np.zeros(N))
        A = cvx.matrix(A)
        b = cvx.matrix(0.0)

        solution = cvx.solvers.qp(P, q, G, h, A, b)
        alpha = np.array(solution["x"])
        return alpha


def main(filename):
    df = pd.read_csv(filename, delimiter=';', low_memory=False)
    discretization(df)
    X = df.drop(columns=['quality']).copy()
    Y = df['quality']

    # split data into train set 0.8, test set 0.1 and validation set 0.1
    X_train, X_rem, Y_train, Y_rem = train_test_split(X,Y, train_size=0.8)
    X_valid, X_test, Y_valid, Y_test = train_test_split(X_rem, Y_rem, test_size=0.5)

    kernel_function = partial(gausian_kernel, gamma=5)

    svm = SVM(X_train.to_numpy(), Y_train.to_numpy(), 1, kernel_function)
    alpha = svm.train()

    print(alpha)
    # Y_pred = []
    # for x in X_valid.to_numpy():
    #     Y_pred.append(np.sign()) #zmienic

    # print(accuracy_score(Y_valid, Y_pred))


if __name__=="__main__":
    main('/home/julia/PAP/pap/zadanie4/winequality-red.csv')