# Zaimplementuj naiwny klasyfikator Bayesa oraz zbadaj działanie algorytmu w zastosowaniu do zbioru danych Iris Data Set.
# Pamiętaj, aby podzielić zbiór danych na zbiór trenujący oraz uczący.

from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


#P(Y=y)
def prior_probabilities(df):
    prior_probs = df.groupby(by = 'type').apply(lambda y : len(y)/len(df))
    return prior_probs

def mean_and_variance(df):
    mean = df.groupby(by = 'type').apply(lambda y: y.mean(axis=0))
    variance = df.groupby(by = 'type').apply(lambda y: y.var(axis=0))
    return mean.values, variance.values

#P(X=x|Y=y)
def gaussian_probabilities(mean, variance, x):
    return (1/np.sqrt(2*np.pi*variance))*np.exp(-((x-mean)**2/(2*variance)))

def posterior_probabilities(row, mean, variance, types, samples):
    posterior_probs = []
    for i in range(len(types)):
        p = 0

        for j in range(samples):
            p += np.log(gaussian_probabilities(mean[i][j], variance[i][j], row[j]))
        posterior_probs.append(p)

    return posterior_probs


def naive_bayes_gaussian(df, X_test):
    samples = len(df.columns)-1
    types = sorted(df['type'].unique())

    mean, variance = mean_and_variance(df)
    prior_probs = prior_probabilities(df)

    Y_pred = []
    for i in range(len(X_test)):
        posterior = posterior_probabilities(X_test[i], mean, variance, types, samples)
        probs = prior_probs + posterior

        max_idx = np.argmax(probs)

        Y_pred.append(max_idx)

    return np.array(Y_pred)


def main():
    filename = '/home/julia/PAP/pap/zadanie7/iris.csv'
    df = pd.read_csv(filename, delimiter=',', low_memory=False)

    train, test = train_test_split(df, test_size=0.2, random_state=22)
    X_test = test.drop(columns=['type']).copy()

    Y_pred = naive_bayes_gaussian(train, X_test.to_numpy())
    Y_test = test['type'].to_numpy()

    accuracy_score = len(test.loc[Y_pred == test['type']])/len(test) * 100
    print(accuracy_score)



if __name__=="__main__":
    main()