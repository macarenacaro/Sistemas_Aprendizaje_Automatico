#**************************** INSPECCIÓN DEL DATASET ****************************
import pandas as pd
import numpy as np

#**************************** 1. Cargar Datos:****************************
df = pd.read_csv('adult.csv', header=0)
print("COLUMNAS:", df.columns) #Nombre de cada column

#**************************** 2. Mostrar datos:****************************
print("DATA-SHAPE: ",df.shape) #Cuantos datos existen (nº filas, nºcolumn)
print(df.head(10)) #muestra 10 filas

print("INFORMACIÓN",df.info()) # Muestra informacion de datos
# por ej columna AGE:   age  32559 non-null (le falta 1 valor para 32560)  float64


print("DESCRIBE:",df.describe()) #df.describe(include='all')
#count: Indica cuántos valores existen en cada columna
#mean: Es el promedio de los valores en cada columna.
#std: Es la desviación estándar, una medida de cuánto se dispersan los valores alrededor del promedio. Un valor más alto significa que los datos están más dispersos. 
# Por ejemplo, capital-gain tiene una desviación estándar alta (7385.4), lo que sugiere gran variación en los valores de esta columna.
#min: Es el valor mínimo encontrado en cada columna. En age, el valor mínimo es 17.
#25%: Es el primer cuartil, es decir, el valor en el que el 25% de los datos son menores. En age, el 25% de las personas tienen 28 años o menos.
#50%: Es la mediana o el segundo cuartil, el valor que está justo en el medio de todos los datos. En age, la mitad de las personas tienen 37 años o menos.
#75%: Es el tercer cuartil, donde el 75% de los valores son menores. En age, el 75% de las personas tienen 48 años o menos.
#max: Es el valor máximo encontrado en cada columna. Para capital-gain, el valor máximo es 99,999, lo que indica un valor de ganancia de capital excepcionalmente alto.

#**************************** 3. Data Prep:****************************
#Para realizar un analisis más completo
#pip install dataprep

from dataprep.eda import create_report

from sklearn.datasets import load_wine

wine = load_wine()
df_wine = pd.DataFrame(wine.data, columns=wine.feature_names)
df_wine_target = pd.DataFrame(wine.target, columns = ['categoria'])

df_wine = df_wine.join(df_wine_target)

#df_wine.sample(5)

create_report(df_wine)