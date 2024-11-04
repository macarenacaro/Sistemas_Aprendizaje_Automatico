import numpy as np
import pandas as pd
from sklearn import utils
from sklearn import preprocessing
from matplotlib import pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

#**************************** 1. Cargar Datos WINE:****************************
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.DataFrame(wine.target)
y.columns = ['class']

#**************************** 2. Revisar la forma de los datos****************************
X.shape # 178 muestras y 13 columnas

#**************************** 3. Creamos un dataframe agregando X e y ****************************
df_wine = pd.concat([X, y], axis=1)

#**************************** 4. Muestra aleatoria de 10 datos del dataframe ****************************
df_wine.sample(10)

#**************************** 5. Ver las clases únicas ****************************
df_wine['class'].unique()

# Procesado de los datos -> en este caso simplemente estandarizamos
sc = StandardScaler()
X_std = sc.fit_transform(X)

X_std

xnew=xi−x¯¯¯σ 

Hace que la media sea 0 y la desviación típica 1.

Existe otros escaladores como

MinMaxScaller: xnew=xi−min(x)max(x)−min(x)
RobustScaler: xnew=xi−min(x)IQR. Donde IQR = InterQuartile Range (Q3-Q1) donde están la mitad de las muestras. Este método se suele usar cuando hay valores atípicos (outlayers) y no deseamos eliminarlos.

Estandarización:

Su utilidad reside en el hecho de que si tenemos dos variables y una tiene valores entre 0 y 3000 y la otra entre 0 y 4, el modelo le dará más importancia a la primera. Depende del modelo en tanto que hay modelos que no tienen esta problemática, en sklearn por ejemplo nos lo indica cuando es obligatorio.
En este caso porque es tan importante... debido a que aplicaremos PCA vamos a ver como es un modelo que define sus componentes según la dirección de mayor varianza, por lo tanto no nos interesa un desajuste de las medidas.
Debería hacerse antes de: PCA / Clustering / KNN / SVM / LASSO / RIDGE. Y en general, para cualquier algoritmo excepto los árboles de decisión. No siempre se obtienen mejoras en Regresión Lineal.

División en dataset de entrenamiento y test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

ó
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

X_train_std
