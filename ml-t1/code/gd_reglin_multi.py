import numpy as np
from Parte2.custo_reglin_multi import custo_reglin_multi


def gd(X, y, alpha, epochs, theta=np.array([0,0,0], ndmin = 2).T):

    m = len(y)

    cost = np.zeros(epochs)

    for i in range(epochs):
        h = X.dot(theta)
        loss = h - y
        gradient = X.T.dot(loss) / m
        theta = theta - (alpha * gradient)
        cost[i] = custo_reglin_multi(X, y, theta=theta)

    return cost[-1], theta
