import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error,r2_score

#***************** 1. Generar datos Aleatorios:**********************
n = 100
x = 6 * np.random.rand(n, 1) - 3  # Datos de entrada
y = 0.5 * x**2 + x + 2 + np.random.randn(n, 1)  # Salida (con algo de ruido)

#*****************1.1. Graficación de los puntos *****************
plt.scatter(x, y)
plt.title("Datos Aleatorios")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

#***************** 2. Ajustar el modelo usando SGD **********************
lin_reg_SGD = SGDRegressor(max_iter=1000, tol=1e-3)  # max_iter define el número de iteraciones
#lin_reg = LinearRegression() #Este seria para Descenso Gradiente (Regresión Lineal)


#***************** 3. Entrenar modelo usando SGD **********************
lin_reg_SGD.fit(x, y.ravel())  # y.ravel() convierte y a un array de 1D
#lin_reg.fit(x, y)#Este seria para Descenso Gradiente (Regresión Lineal)


#***************** 4. Hacer Predicciones **********************
y_pred = lin_reg_SGD.predict(x)
#y_pred = lin_reg.predict(x)#Este seria para Descenso Gradiente (Regresión Lineal)


#*****************5. FUNCION DE COSTE: MSE *****************
# Calcular el MSE
mse = mean_squared_error(y, y_pred)
print("Mean Squared Error (MSE):", mse)
#print(mean_squared_error(y, y_pred)) # Este seria para Descenso Gradiente (Regresión Lineal) LA MISMA


#*****************5. Evaluar Modelo: Coeficiente de Determinacion*****************
print("R^2 Score:", r2_score(y, y_pred))  # Calcula el coeficiente de determinación
#print("R^2 Score (GD):", r2_score(y, y_pred))#Este seria para Descenso Gradiente (Regresión Lineal)


#*****************6. Graficar Resultados **********************
plt.scatter(x, y, label="Datos Reales")
plt.plot(x, y_pred, color='red', label="Predicciones SGD")
plt.title("Predicciones con SGD")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
