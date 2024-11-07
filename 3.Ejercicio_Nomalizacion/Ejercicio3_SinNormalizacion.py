import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics

# 1. Importar Archivo de datos
df = pd.read_csv('HousingData.csv')

# 2. Eliminar filas con valores faltantes
cdf = df.dropna()

# 3. Separar la variable objetivo (MEDV) de las características
target = cdf.iloc[:, -1]  # Selecciona la última columna (MEDV)
df = cdf.iloc[:, 0:13]  # Selecciona las primeras 13 columnas (características)

# 4. Variables X e Y
X = df.to_numpy()
y = target.to_numpy()

#print (X)
#print (y)

#5. Divide el dataset en entrenamiento y test. Usa el 30% para test.
#Utiliza random_state para poder ejecutar varias veces y obtener los mismos datos aleatorios y poder comparar'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# 6. Crea el modelo de Regresión lineal y entrénalo con el conjunto de entrenamiento
regr = linear_model.LinearRegression()  # Crear el modelo de regresión lineal
regr.fit(X_train, y_train)  # Entrenar el modelo con los datos divididos anteriormente


# 7. Calcula y_pred del conjunto de test : Prueba de los datos
y_pred = regr.predict(X_test)  # Hace una predicción en el conjunto de prueba


#8. Muestra los valores de R2 y MSE para el conjunto de test 
print("R2 :\n",metrics.r2_score(y_test, y_pred)) #evaluamos r2
print('MSE:\n', metrics.mean_squared_error(y_test, y_pred))

#9.  Imprimir los coeficientes y el intercepto Generados por el entreamiento 
print('Coefficientes:\n ', regr.coef_)
print('Intercept: \n', regr.intercept_)