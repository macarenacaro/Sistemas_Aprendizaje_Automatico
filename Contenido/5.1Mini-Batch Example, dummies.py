import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor

# Ejemplos de datos
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])  # Entradas
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # Salidas

# ******************** Modelo Mini-Batch SGD
mini_batch_model = SGDRegressor(max_iter=1000, tol=1e-3) 
#. A medida que entrenas el modelo, va ajustando los parámetros para hacer que la predicción sea más cercana a 
# los valores reales (y).

# ********************Crear mini-lotes de tamaño 5
mini_batches = [X[i:i + 5] for i in range(0, len(X), 5)]
mini_batches_y = [y[i:i + 5] for i in range(0, len(y), 5)]

# Entrenar el modelo con los mini-lotes
for x_batch, y_batch in zip(mini_batches, mini_batches_y):
    mini_batch_model.partial_fit(x_batch, y_batch)  # Ajustar el modelo con cada mini-lote

# Imprimir los coeficientes
print("Coeficientes Mini-Batch SGD:", mini_batch_model.coef_)

# Hacer predicciones sobre el conjunto completo de datos
y_pred = mini_batch_model.predict(X)

# Graficar los datos originales y la línea ajustada por el modelo
plt.scatter(X, y, color='blue', label='Datos originales')  # Puntos de datos originales
plt.plot(X, y_pred, color='red', label='Línea ajustada (predicción)')  # Línea ajustada por el modelo
plt.title("Regresión Lineal con Mini-Lotes (SGD)")
plt.xlabel("X (Entradas)")
plt.ylabel("y (Salidas)")
plt.legend()
plt.show()
