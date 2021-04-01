 import pandas as pd

 def get_features_and_target(filepath):
    data = pd.read_csv(filepath, header=None)

    # separa o matriz de dados X (caracteristicas) e o vetor de respostas y (alvo)
    cols = data.shape[1]
    X = data.iloc[:,0:cols-1]  
    y = data.iloc[:,cols-1:cols]
    
    # converte os valores em numpy arrays
    X = np.array(X.values)  
    y = np.array(y.values)
    
    return X,y