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
print(df.head(10)) #VER TABLA 
print(target.head(10)) #VER MEDV


# 4. Convertir DataFrames a matrices NumPy
x = df.to_numpy()  # Características
y = target.to_numpy()  # Objetivo (MEDV)




# 5. Visualizar la relación entre 'AGE' y 'MEDV'
plt.scatter(df['AGE'], y, color='blue')
plt.xlabel("AGE")
plt.ylabel("MEDV")
plt.title("Relación AGE vs MEDV")
plt.show()

# 6. Dividir el conjunto de datos en entrenamiento (80%) y prueba (20%)
msk = np.random.rand(len(cdf)) < 0.8  # Crea una máscara aleatoria
train = cdf[msk]  # Conjunto de entrenamiento (80% de los datos)
test = cdf[~msk]  # Conjunto de test (20% restante de los datos)

# 7. Entrenamiento del modelo
regr = linear_model.LinearRegression()  # Creamos el modelo de regresión lineal
X_train = train[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']]
y_train = train['MEDV']  # Variable dependiente
regr.fit(X_train, y_train)  # Entrenar el modelo con los datos de entrenamiento

# 8. Imprimir los coeficientes y el intercepto del modelo
print('Coeficientes: ', regr.coef_)
print('Intercepto: ', regr.intercept_)

# 9. Hacer predicciones con el conjunto de entrenamiento
y_pred_train = regr.predict(X_train)

# 10. Evaluar el rendimiento del modelo en el conjunto de entrenamiento
train_score = metrics.r2_score(y_train, y_pred_train)
print("R² (entrenamiento):", train_score)

# 11. Hacer predicciones con el conjunto de TEST
X_test = test[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']]
y_test = test['MEDV']
y_pred = regr.predict(X_test)  # Hacer predicciones con los datos de prueba

# 12. Evaluar el rendimiento del modelo en el conjunto de prueba
test_score = metrics.r2_score(y_test, y_pred)
print("R² (prueba):", test_score)

# 13. Ejemplo de predicción con nuevos datos
p = np.array([[0.1, 20, 5, 0, 0.5, 6, 50, 5, 1, 300, 15, 396, 5]])  # Ejemplo de entrada
p_pred = regr.predict(p)  # Predicción
print("Predicción para el nuevo ejemplo:", p_pred)
print("Y_PRED", y_pred)
