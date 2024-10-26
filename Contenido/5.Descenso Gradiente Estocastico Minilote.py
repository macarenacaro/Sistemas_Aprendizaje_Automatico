import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics


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

#******************1.2. Ordenar los datos de entrada******************
x_sorted = np.sort(x, axis=0)  # Ordenar los valores de x para graficar mejor luego

# 2. Generar las características polinómicas (X_poly)
poly_features = PolynomialFeatures(degree=2, include_bias=False)  # Grado 2 para una parábola
X_poly = poly_features.fit_transform(x_sorted)  # Transforma x en [x, x^2]

#***************** 3. Generar mini-lotes aleatorios:**********************
def _get_batch(X, y, batch_size):
    indexes = np.random.randint(len(X), size=batch_size)
    return X[indexes, :], y[indexes, :]

#***************** 4. Crear el modelo:**********************
lin_reg_SGD = SGDRegressor()

#***************** 5. Entrenamiento con mini-lotes:**********************
for i in range(1000):
    XX, YY = _get_batch(X_poly, y, batch_size=10)  # Tomar un mini-lote de tamaño 10
    lin_reg_SGD.partial_fit(XX, YY.ravel())  # Entrenar con el mini-lote

# Ver los coeficientes y el punto de corte
print("Coeficientes:", lin_reg_SGD.coef_)
print("Punto de corte (intercept):", lin_reg_SGD.intercept_)

# Hacer predicciones
y_pred = lin_reg_SGD.predict(X_poly)

# Evaluar las predicciones con R²
print("R² score:", metrics.r2_score(y, y_pred))

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y, y_pred)
print("Mean Squared Error:", mse)


# Gráfico de las predicciones frente a los datos originales
plt.scatter(x_sorted, y)
plt.plot(x_sorted, y_pred, c='r')
plt.show()
