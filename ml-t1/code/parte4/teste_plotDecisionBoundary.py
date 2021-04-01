import scipy.optimize as opt
import pandas as pd

# Carrega os dados do dataset e armazendo em um array.
data = pd.read_csv(filepath, header=None, names=['Teste 1', 'Teste 2', 'Aceito'])  
data.head() 

filepath = "https://raw.githubusercontent.com/MLRG-CEFET-RJ/ml-class/master/ml-t1/data/ex2data2.txt"

plot_microship_data(data)

_lambda = 1
theta = np.zeros((X.shape[1])) #inicialização
result = opt.fmin_tnc(func=custo_reglog_reg, x0=theta, fprime=gd_reglog_reg, args=(X, y, _lambda))
theta = result[0]
J = custo_reglog_reg(theta, X, y, _lambda)
print('Vetor de parâmetros = ', theta)
print('\nCusto = ', J)

plotDecisionBoundary(theta, X, y)
plt.xlabel('Microship teste 1')
plt.ylabel('Microship teste 2')
plt.legend(['y = 1', 'y = 0'])
plt.grid(False)
plt.title('lambda = %0.2f' % _lambda)