"""PASOS
1.Cargar los datos.
2.Graficar los puntos de datos originales.
3.Realizar el gradiente descendente para encontrar los mejores valores de 𝑚 y 𝑏
4.Graficas la línea de regresión resultante.
5.Calculas el error del modelo usando MSE.
6.Calculas el coeficiente de determinación 𝑅2"""

import numpy as np #Biblioteca para manejar arreglos y realizar operaciones matemáticas.
import pandas as pd #Biblioteca para manejar y procesar datos tabulares como los CSV
import matplotlib.pyplot as plt #Biblioteca para graficar datos.

#***************** 1. Procesamiento de los datos de entrada:**********************

#df = pd.read_csv('https://drive.google.com/uc?export=download&id=1fenMPzvOrgT9qoI80tmPAtfbOZYZ3n-X', header=None) desde URL
df = pd.read_csv('data.csv', header=None)
x = df.iloc[:, 0] #Selecciona todas las filas (:) de la primera columna (0).
y = df.iloc[:, 1] #Selecciona todas las filas de la segunda columna.

# Si queremos etiquetar las columnas (features)
df.columns = ['x', 'y']
df.head(5)


# *****************2. Graficación de los datos Reales *****************

#Import de plt: se realizó al inicio
plt.figure(figsize=(12,9))
plt.scatter(x, y)#Muestra los puntos de datos reales en el gráfico.
plt.show()

# *****************3. Gradient Descent: Mejor dato para m y b *****************

#esto sacará cual es el mejor valor de m y b, para el modelo lineal
m = 0 #incializacion de m  
b = 0 #inicialización de b
L = 0.00001  # El ratio de aprendizaje (Learning rate)
epochs = 1000  # Número de iteraciones para hacer el gradient descent
n = float(len(x)) # Número de elementos en x

# Iterando para hacer el Gradient Descent
for i in range(epochs):
   y_pred = m * x + b  # El valor predicho de y actual
   D_m = (-2/n) * sum(x * (y - y_pred))  # Derivada con respecto a m, ve el error también aquí.
   D_b = (-2/n) * sum(y - y_pred)  # Derivada con respecto b, ve el error también aquí.
   m = m - L * D_m  # Actualizar m
   b = b - L * D_b  # Actualizar b

print (m, b)

#*****************4. Gráfica de la línea de regresión*****************

y_pred = m*x + b #Se utiliza la m y b del gradiente anterior

plt.figure(figsize=(12,9))
plt.scatter(x, y) # Muestra los puntos de datos originales reales
plt.plot([min(x), max(x)], [min(y_pred), max(y_pred)], color='red')  # Línea de regresión
#elige los dos puntos de x, el menos y mayor, y el correspondiente y_pred
plt.show()


#*****************5. Función de coste (MSE)*****************

#Evaluacion del margen de error
MSE = np.square(y - y_pred).mean()# Calcula (𝑦−𝑦^)2  Y .mean() para obtener el promedio de todos los errores
print("MSE =", MSE)

#*****************6. Coeficiente de Determinacion*****************

#Evaluacion del modelo
np.corrcoef(x, y)#¿Que es eso?
R2 = np.corrcoef(x, y)[0, 1]**2
print("R2 =", R2)