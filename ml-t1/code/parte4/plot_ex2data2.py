import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt 
import os

import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt 
import os

def plot_microship_data(data):
  # Gera o gráfico de dispersão para análise preliminar dos dados
  positivo = data[data['Aceito'].isin([1])]  
  negativo = data[data['Aceito'].isin([0])]

  fig, ax = plt.subplots(figsize=(8,5))  
  ax.scatter(positivo['Teste 1'], positivo['Teste 2'], s=50, c='k', marker='+', label='Aceito (y = 1)')  
  ax.scatter(negativo['Teste 1'], negativo['Teste 2'], s=50, c='y', marker='o', label='Rejeitado (y = 0)')  
  ax.legend()  
  ax.set_xlabel('Microchip Teste 1')  
  ax.set_ylabel('Microchip Teste 2')
