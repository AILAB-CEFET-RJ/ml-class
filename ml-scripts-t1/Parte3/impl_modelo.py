import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt


def sigmoide(z):
    return 1 / (1 + np.exp(-z))

def funcaoCustoRegressaoLogistica(theta,X,y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    grad0 = np.multiply(-y, np.log(sigmoide(X * theta.T)))
    grad1 = np.multiply((1 - y), np.log(1 - sigmoide(X * theta.T)))
    return np.sum(grad0 - grad1) / (len(X))

def gradiente_descendente(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parametros = int(theta.ravel().shape[1])
    grad = np.zeros(parametros)

    erro = sigmoide(X * theta.T) - y

    for i in range(parametros):
        term = np.multiply(erro, X[:,i])
        grad[i] = np.sum(term) / len(X)

    return grad

def predizer(theta, X):
    probabilidade = sigmoide(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probabilidade]

data = pd.read_csv('../data/ex2data1.txt', header=None, names=['Prova 1', 'Prova 2', 'Aprovado'])
data.head()

data.insert(0, 'Ones', 1)

# converte de dataframes para arrays
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]

# converte de arrays para matrizes
X = np.array(X.values)
y = np.array(y.values)
theta = np.zeros(3)

# gerando o grafico

positivo = data[data['Aprovado'].isin([1])]
negativo = data[data['Aprovado'].isin([0])]

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(positivo['Prova 1'], positivo['Prova 2'], s=50, c='k', marker='+', label='Aprovado')
ax.scatter(negativo['Prova 1'], negativo['Prova 2'], s=50, c='y', marker='o', label='Nao Aprovado')
ax.legend()
ax.set_xlabel('Nota da Prova 1')
ax.set_ylabel('Nota da Prova 2')

funcaoCustoRegressaoLogistica(theta, X, y)

result = opt.fmin_tnc(func=funcaoCustoRegressaoLogistica, x0=theta, fprime=gradiente_descendente, args=(X, y))
funcaoCustoRegressaoLogistica(result[0], X, y)

theta_min = np.matrix(result[0])
x1 = np.array([[1.0,45.0,85.0]])
p = predizer(theta_min, x1)

probabilidade = sigmoide(x1 * theta_min.T)
print("Probabilidade: ", probabilidade[0,0])

theta_min = np.matrix(result[0])
predicoes = predizer(theta_min, X)
corretas = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predicoes, y)]
precisao = (sum(map(int, corretas)) % len(corretas))
print('Acuracia {0}%'.format(precisao))
