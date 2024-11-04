import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import LinearRegression #para x e y
from sklearn import metrics
import matplotlib.pyplot as plt

# 1. Importar Archivo de datos
df = pd.read_csv('HousingData.csv')

# 2. Eliminar filas con valores faltantes
cdf = df.dropna()

# 3. Separar la variable objetivo (MEDV) de las características
target = cdf.iloc[:, -1]  # Selecciona la última columna (MEDV)
df = cdf.iloc[:, 0:13]  # Selecciona las primeras 13 columnas (características)


# 4.1 Entrenamiento del modelo usando X e y
X = df.to_numpy()
y = target.to_numpy()
model = LinearRegression()
model.fit(X, y)

# 4.2 Entrenamiento del modelo usando df y target
regr = linear_model.LinearRegression()  # Creamos el modelo de regresión lineal
regr.fit(df, target)  # Entrenar el modelo con df y target

# 5. Imprimir los coeficientes y el intercepto del modelo
print('Coeficientes: ', regr.coef_)
print('Intercepto: ', regr.intercept_)

# 6. Calcular el coeficiente de determinación R²
r2_score = metrics.r2_score(target, regr.predict(df))
print("R² (total):", r2_score)

# 7. Predicción para la vivienda con los datos proporcionados
p = np.array([[0.03731, 0.0, 7.07, 0.0, 0.469, 6.421, 23.5, 4.9671, 2.0, 242.0, 17.8, 396.90, 9.14]])  # Datos de entrada
p_pred = regr.predict(p)  # Predicción
print("Predicción para el nuevo ejemplo:", p_pred)

# 8. Añadido de YAPA :) Calcular el coeficiente de determinación R²
y_pred = regr.predict(X)
print('MSE:', metrics.mean_squared_error(y, y_pred))