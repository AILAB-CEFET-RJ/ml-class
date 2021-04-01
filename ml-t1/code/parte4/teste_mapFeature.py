filepath = "https://raw.githubusercontent.com/MLRG-CEFET-RJ/ml-class/master/ml-t1/data/ex2data2.txt"
X = mapFeature(X[:, 0], X[:, 1])
print(X.shape) # esperado: (118, 28)