import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
import matplotlib.mlab as mlab
from scipy.io import loadmat  
from scipy import stats 

# Source: https://github.com/Grzego/handwriting-generation/issues/16 
def bivariate_normal(X, Y, sigmax=1.0, sigmay=1.0,
                     mux=0.0, muy=0.0, sigmaxy=0.0):
    """
    Bivariate Gaussian distribution for equal shape *X*, *Y*.
    See `bivariate normal
    <http://mathworld.wolfram.com/BivariateNormalDistribution.html>`_
    at mathworld.
    """
    Xmu = X-mux
    Ymu = Y-muy

    rho = sigmaxy/(sigmax*sigmay)
    z = Xmu**2/sigmax**2 + Ymu**2/sigmay**2 - 2*rho*Xmu*Ymu/(sigmax*sigmay)
    denom = 2*np.pi*sigmax*sigmay*np.sqrt(1-rho**2)
    return np.exp(-z/(2*(1-rho**2))) / denom

def estimate_gaussian_params(X):  
	########################
	# SEU CODIGO AQUI :
	# Essa funcao deve computar e retornar mu e sigma2,
	# vetores que contem a media e a variancia de cada
	# caracteristica (feature) de X.
	########################
    return (mu, sigma2)

def select_epsilon(pval, yval):  
    best_epsilon_value = 0
    best_f1_value = 0

    step_size = (pval.max() - pval.min()) / 1000

    print('step size: ' + str(step_size))

    for epsilon in np.arange(pval.min(), pval.max(), step_size):
        preds = pval < epsilon
		########################
		# SEU CODIGO AQUI :
		# Dentro deste loop, voce deve implementar logica para
		# definir corretamente os valores das variaveis
		# best_epsilon_value e best_f1_value.
		########################
    return best_epsilon_value, best_f1_value

def main():
	data = loadmat('../data/ex8data1.mat') 
	X = data['X']  

	(mu, sigma2) = estimate_gaussian_params(X)
	print('mu: ' + str(mu))
	print('variance: ' + str(sigma2))

	# Plot dataset
	plt.scatter(X[:,0], X[:,1], marker='x')  
	plt.axis('equal')
	plt.show()

	# Plot dataset and contour lines
	plt.scatter(X[:,0], X[:,1], marker='x')  
	x = np.arange(0, 25, .025)
	y = np.arange(0, 25, .025)
	first_axis, second_axis = np.meshgrid(x, y)
	Z = bivariate_normal(first_axis, second_axis, np.sqrt(sigma2[0]), np.sqrt(sigma2[1]), mu[0], mu[1])
	plt.contour(first_axis, second_axis, Z, 10, cmap=plt.cm.jet)
	plt.axis('equal')
	plt.show()

	# Load validation dataset
	Xval = data['Xval']  
	yval = data['yval'].flatten()

	stddev = np.sqrt(sigma2)

	pval = np.zeros((Xval.shape[0], Xval.shape[1]))  
	pval[:,0] = stats.norm.pdf(Xval[:,0], mu[0], stddev[0])  
	pval[:,1] = stats.norm.pdf(Xval[:,1], mu[1], stddev[1])  
	print(np.prod(pval, axis=1).shape)
	epsilon, _ = select_epsilon(np.prod(pval, axis=1), yval)  
	print('Best value found for epsilon: ' + str(epsilon))

	# Computando a densidade de probabilidade 
	# de cada um dos valores do dataset em 
	# relacao a distribuicao gaussiana
	p = np.zeros((X.shape[0], X.shape[1]))  
	p[:,0] = stats.norm.pdf(X[:,0], mu[0], stddev[0])  
	p[:,1] = stats.norm.pdf(X[:,1], mu[1], stddev[1])

	# Apply model to detect abnormal examples in X
	anomalies = np.where(np.prod(p, axis=1) < epsilon)

	# Plot the dataset X again, this time highlighting the abnormal examples.
	plt.clf()
	plt.scatter(X[:,0], X[:,1], marker='x')  
	plt.scatter(X[anomalies[0],0], X[anomalies[0],1], s=50, color='r', marker='x')  
	plt.axis('equal')
	plt.show()

if __name__ == "__main__":
	main()
