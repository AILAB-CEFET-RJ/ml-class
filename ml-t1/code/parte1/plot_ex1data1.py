import os
import numpy as np
import matplotlib.pyplot as plt

def plot(X, y):
    plt.scatter(X.T, y, color='red', marker='x')
    plt.title('População da cidade x Lucro da filial')
    plt.xlabel('População da cidade (10k)')
    plt.ylabel('Lucro (10k)')

    filename = 'target/plot1.1.png'
    if not os.path.exists(os.path.dirname(filename)):
      os.makedirs(os.path.dirname(filename))

    plt.savefig(filename)
    plt.show()