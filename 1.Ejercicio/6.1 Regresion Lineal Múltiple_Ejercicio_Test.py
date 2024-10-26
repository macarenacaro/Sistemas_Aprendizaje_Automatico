import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import metrics
import matplotlib.pyplot as plt

# 1. Importar Archivo de datos
df = pd.read_csv('HousingData.csv')

# 2. Eliminar filas con valores faltantes
cdf = df.dropna()

# 3. Separar la variable objetivo (MEDV) de las características
target = cdf.iloc[:, -1]  # Selecciona la última columna (MEDV)
df = cdf.iloc[:, 0:13]  # Selecciona las primeras 13 columnas (características)

# 4. Establecer parametros % de test y de Entrenamient **********************
# Crear una máscara booleana para dividir los datos en entrenamiento (80%) y prueba (20%)
msk = np.random.rand(len(cdf)) < 0.8 
train = cdf[msk]# Conjunto de entrenamiento
test = cdf[~msk]# Conjunto de test (resto)


#5.****************** Entrenamiento 
regr = linear_model.LinearRegression()  # Crear el modelo de regresión lineal
X_train = train[['CRIM',	'ZN',	'INDUS',	'CHAS',	'NOX',	'RM',	'AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']] # Variables independientes
y_train = train[['MEDV']] # Variables dependientes
regr.fit(X_train, y_train)  # Entrenar el modelo

#5.1  Imprimir los coeficientes y el intercepto Generados por el entreamiento (No hace grafica, es como un esquema)
print('Coefficients: ', regr.coef_)
print('Intercept: ', regr.intercept_)

#5.1  R2
y_pred_train = regr.predict(X_train)
metrics.r2_score(y_train, y_pred_train)

#***************** 6. Prueba **********************
X_test = test[['CRIM',	'ZN',	'INDUS',	'CHAS',	'NOX',	'RM',	'AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']]
y_test = test[['MEDV']]
y_pred = regr.predict(X_test) #Hace una prediccion de la prueba??
metrics.r2_score(y_test, y_pred) #evaluamos r2


# 7. Predicción para la vivienda con los datos proporcionados
p = np.array([[0.03731, 0.0, 7.07, 0.0, 0.469, 6.421, 23.5, 4.9671, 2.0, 242.0, 17.8, 396.90, 9.14]])  # Datos de entrada
p_pred = regr.predict(p)  # Predicción
print("Predicción para el nuevo ejemplo:", p_pred)
print(y_pred)