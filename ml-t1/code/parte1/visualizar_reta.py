import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


def plot(X, y, theta):
    t = np.arange(0, 25, 1)
    plt.scatter(X, y, color='red', marker='x', label='Dados de treinamento')
    plt.plot(t, theta[0] + (theta[1]*t), color='blue', label='Regressão Linear')
    plt.axis([4, 25, -5, 25])
    plt.title('Populacao da cidade x Lucro da filial')
    plt.xlabel('Populacao da cidade (10k)')
    plt.ylabel('Lucro (10k)')
    plt.legend()
    plt.show()

    filename = 'target/plot1.2.png'
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    plt.savefig(filename)
