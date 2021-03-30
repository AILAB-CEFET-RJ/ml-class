def costFunctionReg(theta, X, y, lambda_):
  m = y.size #número de interações
  J = 0
  grad = np.zeros(theta.shape)
  h = sigmoide(X.dot(theta.T))
  temp = theta
  temp[0] = 0
  J = (1 / m) * np.sum(-y.dot(np.log(h)) - (1 - y).dot(np.log(1 - h))) + (lambda_ / (2 * m)) * np.sum(np.square(temp))
  grad = (1 / m) * (h - y).dot(X)
  grad = grad + (lambda_ / m) * temp
  return J, grad
