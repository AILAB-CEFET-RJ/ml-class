examData_norm, mean_examData, std_examData = normalizar_caracteristica(examData)
examData_norm[:5,:]

examData_norm = np.insert(examData_norm, 0, 1, axis=1)
print(examData_norm[:5,:])

theta = np.array([[0,0,0]]) #inicialização
J = custo_reglog(theta, examData_norm, labels_norm)
print('Custo = ', J)