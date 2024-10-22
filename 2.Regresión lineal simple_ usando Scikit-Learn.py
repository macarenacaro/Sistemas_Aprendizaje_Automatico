import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #Biblioteca para graficar datos.
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score



#***************** 1. Procesamiento de los datos de entrada:**********************

# Otra forma de leer el csv
data = np.genfromtxt('https://drive.google.com/uc?export=download&id=1fenMPzvOrgT9qoI80tmPAtfbOZYZ3n-X', delimiter=',')
x = data[:, 0]
y = data[:, 1]

# Convierte los datos en columnas
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

# *****************2. Graficación de los datos Reales *****************

plt.figure(figsize=(12,9))
plt.scatter(x, y)
plt.show()

# *****************3. Entrenar el modelo: Mejor dato para m y b *****************

reg = LinearRegression()
reg.fit(x, y) # Entrenar el modelo con los datos
m = reg.coef_[0][0] # Obtener la pendiente
print(m)
b = reg.intercept_[0] # Obtener el intercepto
print(b)

#*****************4. OPCION1: Gráfica de la línea de regresión*****************

y_pred = m*x + b # OPCION 1 # Usar la ecuación y = mx + b
plt.figure(figsize=(12,9))
plt.scatter(x, y) # Graficar puntos originales
plt.plot([min(x), max(x)], [min(y_pred), max(y_pred)], color='red')  # Línea de regresión
plt.show()


#*****************4. OPCION2: Gráfica de la línea de regresión*****************

y_pred = reg.predict(x)# OPCION 2 # En vez de: y = mx + b
plt.figure(figsize=(12,9))
plt.scatter(x, y)# Graficar puntos originales
plt.plot([min(x), max(x)], [min(y_pred), max(y_pred)], color='red')  # Línea de regresión
plt.show()


#*****************5.OPCION1: Función de coste (MSE)*****************

MSE = np.square(np.subtract(y,y_pred)).mean() 
#Calcula la diferencia entre los valores reales y y los predichos y_pred.
#np.square(...): Eleva al cuadrado estas diferencias.
#.mean(): Calcula el promedio de los errores al cuadrado
print(MSE)

#*****************5.OPCION2: Función de coste (MSE)*****************
print(mean_squared_error(y, y_pred))

#*****************6. Coeficiente de Determinacion*****************
print(r2_score(y, y_pred))


