import scipy.optimize as opt
from util import get_features_and_target

filepath = 'https://raw.githubusercontent.com/MLRG-CEFET-RJ/ml-class/master/ml-t1/data/ex2data1.txt'

examData, labels = get_features_and_target(filepath)
examData_norm, mean_examData, std_examData = normalizar_caracteristica(examData)
examData_norm[:5,:]

examData_norm = np.insert(examData_norm, 0, 1, axis=1)
print(examData_norm[:5,:])

result = opt.fmin_tnc(func=custo_reglog, x0=theta, fprime=gd_reglog, args=(examData_norm, labels))
theta = result[0]
J = custo_reglog(theta, examData_norm, labels)

print('Vetor de parâmetros = ', theta)
print('Custo = ', J)