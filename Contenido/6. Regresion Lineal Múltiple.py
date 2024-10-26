import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics

#***************** 1. Cargar Datos:**********************
df = pd.read_csv("FuelConsumption.csv")
print(df.head())

#***************** 2. Mostrar datos**********************
# Seleccionamos columnas específicas del DataFrame para una nueva variable
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)

#***************** 3. Seleccionar una Xs e Y **********************
plt.scatter(cdf['ENGINESIZE'], cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

#***************** 4. Establecer parametros % de test y de Entrenamient **********************

# Crear una máscara booleana para dividir los datos en entrenamiento (80%) y prueba (20%)
msk = np.random.rand(len(df)) < 0.8 
train = cdf[msk]# Conjunto de entrenamiento
test = cdf[~msk]# Conjunto de test (resto)
msk = np.random.rand(len(df))


#***************** 5. Entrenamiento **********************

regr = linear_model.LinearRegression()  # Crear el modelo de regresión lineal
X_train = train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']] # Variables independientes
y_train = train[['CO2EMISSIONS']] # Variables dependientes
regr.fit(X_train, y_train)  # Entrenar el modelo, ajusta los coeficientes del modelo para que pueda hacer predicciones precisas.

# Imprimir los coeficientes y el intercepto Generados por el entreamiento (No hace grafica, es como un esquema)
print('Coefficients: ', regr.coef_)
print('Intercept: ', regr.intercept_)

#***************** 6. Predicción **********************
y_pred_train = regr.predict(X_train) # le damos los valores entrenados a nuestra y predicha. 
metrics.r2_score(y_train, y_pred_train) #Se mide cuán bueno es el modelo al predecir las emisiones de CO2.

#***************** 7. Prueba **********************
from sklearn import metrics
X_test = test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']]
y_test = test[['CO2EMISSIONS']]
y_pred = regr.predict(X_test)
metrics.r2_score(y_test, y_pred)

#***************** 7. Datos para que predecir **********************
p = [[2.0, 4, 8.5]]
p_pred = regr.predict(p)
print (p_pred)
print(y_pred)