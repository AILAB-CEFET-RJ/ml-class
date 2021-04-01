def custo_reglog_reg(theta, X, y, _lambda):
    m = len(X)
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    
    grad0 = np.multiply(-y, np.log(sigmoide(X * theta.T)))
    grad1 = np.multiply((1 - y), np.log(1 - sigmoide(X * theta.T)))
    
    # não considera theta0 para o cálculo
    theta_j = theta[:,1:]
    regularizacao = (_lambda / (2 * m)) * np.sum(np.dot(theta_j.T,theta_j))     
    return np.sum((grad0 - grad1) / m) + regularizacao

def gd_reglog_reg(theta, X, y, _lambda):
    m = len(X)
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parametros = int(theta.ravel().shape[1])
    grad = np.zeros(parametros)

    erro = sigmoide(X * theta.T) - y

    for i in range(parametros):
        term = np.multiply(erro, X[:,i])
        if (i != 0):
            regularizacao = ((_lambda / m) * theta[:,i])
            grad[i] = (np.sum(term) / m) + regularizacao
        else:
            grad[i] = np.sum(term) / m 

    return grad