import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler


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


# 6. ANTES DE CREAR MODELO: Se normalizan los datos:
# Normalizamos estas variables porque son las características de ENTRADA del modelo.
scaler = StandardScaler()         
X_train_scaled = scaler.fit_transform(X_train)  # CON FIT: escalador calcula los parámetros necesarios para la normalización, como la media y la desviación estándar de X_train
X_test_scaled = scaler.transform(X_test)        # SIN FIT: scaler.transform(X_test), estás aplicando la normalización a X_test usando los parámetros que calculaste con X_train


#7.Crea el modelo de Regresión lineal y entrénalo con el conjunto de entrenamiento
modelo2 = LinearRegression()
modelo2.fit(X_train_scaled, y_train)  # Entrenar el modelo con los datos normalizados


# 7. Calcula y_pred del conjunto de test : Prueba de los datos
y_pred2 = modelo2.predict(X_test_scaled)  # Predicción con los datos normalizados


#8. Muestra los valores de R2 y MSE para el conjunto de test 
print("Normalizado R2:\n",metrics.r2_score(y_test, y_pred2)) #evaluamos r2
print('Normalizado MSE:\n', metrics.mean_squared_error(y_test, y_pred2))

#9.  Imprimir los coeficientes y el intercepto Generados por el entreamiento 
print('Coefficientes:\n ', modelo2.coef_)
print('Intercept: \n', modelo2.intercept_)