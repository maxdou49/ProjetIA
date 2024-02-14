import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.neural_network import MLPClassifier

import math
import timeit

data = pd.read_csv("mobile_train.csv", delimiter=",")
# print(data)

train = data.sample(frac=0.8)
valid = data.drop(train.index)
# train = data
# valid = pd.read_csv("mobile_test_data.csv", delimiter=",")

X_train = train.iloc[:, :-1]
Y_train = train.iloc[:, -1]
# print(X_train)
# print(Y_train)
# X_valid = valid
X_valid = valid.iloc[:, :-1]
Y_valid = valid.iloc[:, -1]

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean)/std
mean = X_valid.mean(axis=0)
std = X_valid.std(axis=0)
X_valid = (X_valid - mean)/std

def distance(v1, v2):
    return math.sqrt(((v2 - v1) ** 2).sum())


def neighbors(X_train, Y_train, x_test, k):
    list_distance = []
    for i in range(X_train.shape[0]):
        list_distance.append(distance(x_test, X_train.iloc[i]))

    df = pd.DataFrame()
    df["label"] = Y_train
    df["dist"] = list_distance
    df = df.sort_values(by="dist")
    return df.iloc[:k, :]


def prediction(neighbors):
        classes = np.zeros(4)
        for i in range(neighbors.shape[0]):
            classes[int(neighbors.iloc[i]["label"])] += 1

        return classes.argmax()
    # return int(round(np.array(neighbors).mean()))


def k_test(X_train, X, Y_train, Y, k):
    nearest = neighbors(X_train, Y_train, X, k)
    if(prediction(nearest) == Y):
        return 1
    else:
        return 0


def evaluation(X_train, X_valid, Y_train, Y_valid, k, jobs=1):
    res = Parallel(n_jobs=jobs)(delayed(k_test)(X_train, X_valid.iloc[i], Y_train, Y_valid.iloc[i], k) for i in range(X_valid.shape[0]))

    accuracy = np.sum(res)/len(res)
    print(accuracy)
    return accuracy


def tourner_test(X_train, Y_train, X, k):
    return prediction(neighbors(X_train, Y_train, X, k))


def tourner(X_train, Y_train, valid, k, jobs=1):
    res = Parallel(n_jobs=jobs)(
        delayed(tourner_test)(X_train, Y_train, valid.iloc[i], k) for i in range(valid.shape[0]))
    return res

# klist = [25, 27, 29, 31, 33]
# res = []
# for k in klist:
#     print(k)
#     res.append(evaluation(X_train, X_valid, Y_train, Y_valid, k, 7))
#
# plt.plot(klist, res)
# plt.show()

# res = tourner(X_train, Y_train, X_valid, 27, 7)
# np.savetxt("mobile_test_predictions.csv", res, delimiter=",")

# Regression

# def sigmoid(z):
#     return 1/(1+np.exp(-z))
#
# def output(X, w):
#     return sigmoid(np.dot(X, w))
#
# def binary_cross_entropy(f, y):
#     return -(y*np.log(f) + (1-y)*np.log(1 - f)).mean()
#
# def gradient(f, y, X):
#     grad = -np.dot(np.transpose(X),(y-f))/X.shape[0]
#     return grad
#
# def train(X_train, Y_train, eta, nb_iter):
#     w = np.random.randn(X_train.shape[1]) # On initialise le vecteur
#     for i in range(nb_iter): # On fait nb_iter iterations
#         f_train = output(X_train, w) # On regarde le resultat
#         grad = gradient(f_train, Y_train, X_train) # On regarde le gradient
#         w = w - eta*grad # On altère w
#     return w
#
#
# print()

# Utilisation de MLPClassifier

def tourner_mlp(a, l):
    print(str(a) + " " + str(l))
    clf = MLPClassifier(tol=1e-5, hidden_layer_sizes=l, alpha=a, max_iter=5000)
    clf.fit(X_train, Y_train)
    return clf.score(X_valid, Y_valid)

# iter = [1000, 2500, 5000, 10000]
alpha = [1e-3, 3e-3, 5e-3, 1e-2, 3e-2, 4e-2, 5e-2, 6e-2]
# layers = [[[100], [200], [300], [500]], [[100,100], [200, 200], [300, 300], [500,500]], [[100,100,100], [200,200,200], [300,300,300], [500,500,500]]]
layers = [1,2,3,4]
layer_size = [50,100,200,300,500]
acc = []
for l1 in layers:
    accl = []
    for l2 in layer_size:
        l = np.repeat(l2, l1)
        accl.append(Parallel(n_jobs=8)(
            delayed(tourner_mlp)(a, l) for a in alpha))
    # for l in layers:
    #     print(str(i)+" "+str(a)+" "+str(l))
    #     clf = MLPClassifier(tol=1e-5, hidden_layer_sizes=l, alpha=a, max_iter=i)
    #     clf.fit(X_train, Y_train)
    #     acca.append(clf.score(X_valid, Y_valid))
    acc.append(accl)

for i in range(len(layers)):
    print(str(layers[i])+" couches")
    plt.imshow(acc[i])
    plt.show()

# print(acc)



# clf = MLPClassifier(max_iter=1000, hidden_layer_sizes=200)
# clf.fit(X_train, Y_train)
# print("Accuracy" + str(clf.score(X_valid, Y_valid)))