def learningCurve(X, y, Xval, yval, lambda_=0):

    m = y.size

    error_train = np.zeros(m)
    error_val   = np.zeros(m)

    for i in range(1, m + 1):
        theta_t = trainLinearReg(linearRegCostFunction, X[:i], y[:i], lambda_ = lambda_)
        error_train[i - 1], _ = linearRegCostFunction(X[:i], y[:i], theta_t, lambda_ = 0)
        error_val[i - 1], _ = linearRegCostFunction(Xval, yval, theta_t, lambda_ = 0)
        
    return error_train, error_val