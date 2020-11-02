import numpy as np

#Z-Score norm
def normalizar_caracteristica(X, y):
    mean_X = np.mean(X, axis=0)
    std_X = np.std(X, axis=0)
    X_norm = (X - mean_X) / std_X

    mean_y = np.mean(y)
    std_y = np.std(y)
    y_norm = (y - mean_y) / std_y

    # Incluir o valor de 1 em x, pois theta0 = 1
    X_norm = np.c_[np.ones((X_norm.shape[0], 1)), X_norm]

    return X_norm, y_norm
