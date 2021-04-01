import numpy as np
import matplotlib.pyplot as plt
import os

def learningCurve(theta, X, y, Xval, yval, _lambda):
    m = len(X)
    erros_treino = np.zeros(m)
    erros_val = np.zeros(m)
    qtds_exemplos = []
    
    for i in range(1,m+1):
        treino_subset = X[:i,:]
        y_subset = y[:i]
        qtds_exemplos.append(len(treino_subset))
        
        result = encontrar_theta_otimo(theta, treino_subset, y_subset, _lambda)
        theta = result[0]
        
        J_treino = custo_reglin_regularizada(theta, treino_subset, y_subset, _lambda=0)
        J_val = custo_reglin_regularizada(theta, Xval, yval, _lambda)
        
        erros_treino[i-1] = J_treino
        erros_val[i-1] = J_val
    
    return qtds_exemplos, erros_treino, erros_val
