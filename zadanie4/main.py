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
        print("Started training")

        A = self.Y.reshape(1, N)
        A = A.astype('float')

        P = cvx.matrix((np.matmul(self.Y,np.transpose(self.Y)) * self.kernel))
        q = cvx.matrix(-np.ones((N,1)))
        G = cvx.matrix(np.vstack((np.eye(N) * -1, np.eye(N))))
        h = cvx.matrix(np.hstack((np.zeros(N), np.ones(N) * self.C)))
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
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, train_size=0.8)

    kernel_function = partial(gausian_kernel, gamma=15)
    kernel_function = linear_kernel

    svm = SVM(X_train.to_numpy(), Y_train.to_numpy(), 0.005, kernel_function)
    alpha = svm.train()

    print(alpha)
    Y_pred = []
    for x in X_test.to_numpy():
         Y_pred.append(np.sign(np.sum([alpha[i]*Y_train.to_numpy()[i]*kernel_function(X_train.to_numpy()[i], x) for i in range(len(alpha))])))

    print(accuracy_score(Y_test, Y_pred))


if __name__=="__main__":
    main('/home/julia/PAP/pap/zadanie4/winequality-white.csv')