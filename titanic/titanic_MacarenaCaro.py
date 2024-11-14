import numpy as np
import pandas as pd

#**************************** 1. Cargar Datos:*****************
#***** 1. Base de Datos de test y train

df_train = pd.read_csv('train.csv', header = 0, dtype={'Age': np.float64})
df_test  = pd.read_csv('test.csv' , header = 0, dtype={'Age': np.float64})
df = df_train.append(df_test, ignore_index=True)

#df = pd.read_csv('adult.csv', header=0)
print("COLUMNAS:", df.columns) #Nombre de cada column***********

#**************************** 2. Valores Nan:*****************
