import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from Parte1.custo_reglin_uni import custo_reglin_uni
from Parte1.gd_reglin_uni import gd_reglin_uni

dataset = pd.read_csv('../data/ex1data1.txt', header = None)

x = dataset.iloc[:, 0:-1].values
y = dataset.iloc[:, -1:].values

plt.scatter(x, y, color='red', marker='x')
plt.title('Populaçãoo da cidade x Lucro da filial')
plt.xlabel('População da cidade (10k)')
plt.ylabel('Lucro (10k)')
plt.show()

# Incluir o valor de 1 em x, pois theta0 = 1
x = np.c_[np.ones((x.shape[0],1)), x]

# Conforme solicitado no enunciado para iniciar todos os parâmetros com zero
print(custo_reglin_uni(x,y, theta = np.array([0,0], ndmin=2).T))

# Conforme solicitação no texto do trabalho inicie o valor da taxa de aprendizagem com 0.01
cost_final, theta = gd_reglin_uni(x, y, 0.01, 5000)
print(cost_final)

x = dataset.iloc[:, 0:-1].values
t = np.arange(0, 25, 1)
plt.scatter(x, y, color='red', marker='x', label='Training Data')
plt.plot(t, theta[0] + (theta[1]*t), color='blue', label='Linear Regression')
plt.axis([4, 25, -5, 25])
plt.title('População da cidade x Lucro da filial')
plt.xlabel('População da cidade (10k)')
plt.ylabel('Lucro (10k)')
plt.legend()
plt.show()

# Predição para 35.000 habitantes
print(np.array([1, 3.5]).dot(theta))

# Predição para 70.000 habitantes
print(np.array([1, 7]).dot(theta))

# Incluir o valor de 1 em x, pois theta0 =1
x = np.c_[np.ones((x.shape[0],1)), x]

# Valores de theta0 e theta1 informados no enunciado do trabalho
theta0 = np.arange(-10, 10, 0.01)
theta1 = np.arange(-1, 4, 0.01)

# Inicialização dos valores de J com zeros
J = np.zeros((len(theta0), len(theta1)))

# Preenchendo os valores de J
for i in range(len(theta0)):
    for j in range(len(theta1)):
        t = [[theta0[i]], [theta1[j]]]
        J[i,j] = custo_reglin_uni(x, y, t)

# Transpondo J devido as funcoes contour/meshgrid
J = np.transpose(J)

# Plotar a funcoo de custo utilizando levels como logspace. Range -1 ~ 4 devido ao
# range de theta1 e 20 pois o theta0 tem 20 valores (-10 ate 10)
fig = plt.figure()
fig, ax = plt.subplots()
ax.contour(theta0, theta1, J, levels=np.logspace(-1, 4, 20), color='blue')
ax.plot(theta[0,0], theta[1,0], 'rx')
plt.xlabel('theta0')
plt.ylabel('theta1')
plt.show()

# Abaixo segue o codigo desenvolvido para geracao da superficie da funcao J

# Comandos necessarios para o matplotlib plotar em 3D
fig = plt.figure()
ax = fig.gca(projection='3d')

# Plotando o grafico de superficie
theta0, theta1 = np.meshgrid(theta0, theta1)
surf = ax.plot_surface(theta0, theta1, J)
plt.xlabel('theta_0')
plt.ylabel('theta_1')
plt.show()
