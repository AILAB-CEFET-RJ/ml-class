import matplotlib.pyplot as plt
import scipy.io
import requests

def plot_ex5data1(X, y):
    plt.figure(figsize=(8,5))
    plt.xlabel('Mudança no nível da água (x)')
    plt.ylabel('Água saindo da barragem (y)')
    plt.plot(X,y,'rx')

filepath = 'https://github.com/MLRG-CEFET-RJ/ml-class/blob/master/ml-t1/data/ex5data1.mat?raw=true'
r = requests.get(filepath, allow_redirects=True)
open('ex5data1.mat', 'wb').write(r.content)

data = scipy.io.loadmat('ex5data1.mat')

_X, y = data['X'], data['y'] # conjunto de treinamento
_Xval, yval = data['Xval'], data['yval'] # conjunto de desenvolvimento
_Xtest, ytest = data['Xtest'], data['ytest'] # conjunto de teste

plot_ex5data1(_X, y)
