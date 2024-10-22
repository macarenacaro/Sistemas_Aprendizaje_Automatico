"""PASOS
1.Generar datos aleatorios.
1.1.Graficar los puntos.
2.Crear las "features" polinómicas.
3.Entrenar Modelo.
4.Calculas Intercepto (b) y R2 Coeficiente.
5.Hacer predicciones
6.Graficar el ajuste
7."""

import numpy as np
import matplotlib.pyplot as plt

#***************** 1. Generar datos Aleatorios:**********************
# generando un conjunto de datos aleatorio que se ajuste a una curva
# np.random.rand genera un número aleatorio correspondiente a una distribuación
# normal de media =0 y desviación típica = 1
n = 100
x = 6 * np.random.rand(n, 1) - 3
x_sorted = np.sort(x, axis=0)
y = 0.5 * x_sorted**2 + x_sorted + 2 + np.random.randn(n, 1) # randn -> normal distribution


#*****************1.1. Graficación de los puntos *****************
plt.figure(figsize=(12,9))
plt.scatter(x_sorted, y)
plt.show()

#***************** 2. Crear features polinomicas **********************
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2, include_bias=False) #degree = 2 -> agrega x^2
X_poly = poly_features.fit_transform(x_sorted)  # transforma x a x y x^2
print(X_poly.shape) #(100,2) es: 100 filas (n=100) y 2 columnas "features" polinómicas: 𝑥 y x^2
print(X_poly)
"""[ x1, x1^2 ],
[ x2, x2^2 ],
 ...
 [ xn, xn^2 ]]"""

#***************** 3. Entrenar Modelo **********************
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y) #ajusta los parámetros de la regresión lineal a los datos, no se hacen iteraciones
"""Entrenamos el modelo con lin_reg.fit(), lo que significa que encuentra los mejores valores de w0, w1, y w2
para que la curva predicha se ajuste lo mejor posible a los datos."""

#***************** 4. Intercepto y Coficient (m1,m2) **********************
print("Punto de corte:", lin_reg.intercept_) 
print("Coef:", lin_reg.coef_) 

#***************** 5. Predicciones de y **********************
y_pred = lin_reg.predict(X_poly)
#Ahora usamos el modelo para predecir los valores de y usando las "features" polinómicas (x y x^2).

#***************** 6.Graficar el ajuste **********************
plt.figure(figsize=(12,9))
plt.scatter(x_sorted, y)
plt.plot(x_sorted, y_pred, c='r')
plt.show()


#***************** 7.Evaluar **********************
from sklearn import metrics
from sklearn.metrics import mean_squared_error
print(f"r2_score: {metrics.r2_score(y, y_pred)}")
print(f"mean_squared_error: {mean_squared_error(y, y_pred)}")
#El r2_score: qué tan bien se ajusta el modelo a los datos. Cuanto más cercano a 1, mejor es el ajuste.
#El mean_squared_error: promedio de los errores al cuadrado. Cuanto más bajo, mejor.