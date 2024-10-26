import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# 1. Importar Archivo de datos
# Data extracted from https://archive.ics.uci.edu/ml/datasets/Glass+Identification
df_headers = ["Id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "glass-type"]
df = pd.read_csv("glass.data", names = df_headers)

"""Type of glass: (class attribute)
-- 1 building_windows_float_processed
-- 2 building_windows_non_float_processed
-- 3 vehicle_windows_float_processed
-- 4 vehicle_windows_non_float_processed (NO ESTÁ EN LA TABLA)
-- 5 containers
-- 6 tableware
-- 7 headlamps"""

# 2. Mostrar algunos ejemplos del DataFrame aleatorios
print (df.sample(10)) # Selección aleatoria del DataFrame.

#3. Mostrar los tipos de vidrio únicos (las diferentes clasificaciones)
df["glass-type"].unique() #array([1, 2, 3, 5, 6, 7]) #son 6 total mostrados arriba

#4.Clasificar categorías por sus nombres ej: 1 = building_windows_float_processed
categories = ('building_windows_float_processed','building_windows_non_float_processed','vehicle_windows_float_processed','containers','tableware','headlamps')

#5.Gráfico de dispersión: Mostrar gráfica de Na vs glass-type
plt.figure(figsize=(12,9))
scatter = plt.scatter(df.index, df["Na"], c = df["glass-type"], cmap = 'Accent')
plt.legend(handles = scatter.legend_elements()[0],
           labels = categories,
           title  ="glass type")
plt.xlabel("sample")
plt.ylabel("Na")


'''
Separa X e y usando iloc
'''
X = df.iloc[:, 1:10]  # Todas las columnas desde 'RI' hasta 'Fe' (excluyendo 'Id' y 'glass-type')
y = df["glass-type"]  # La columna objetivo es 'glass-type'
print ("VALORES DE X:\n",X)
print ("VALORES DE Y:\n",y)

'''
Busca en internet train_test_split y divide X e y en Entrenamiento y Test: X_train, X_test, y_train, y_test
con 70% para entrenamiento y 30% para test
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#La primera variable: entrenamiento de X.
#La segunda variable: prueba de X.
#La tercera variable: entrenamiento de y.
# La cuarta variable:  prueba de y.
#train_test_split divide X e y en cuatro subconjuntos: X_train, X_test, y_train, y_test.
#test_size=0.3 significa que el 30% de los datos irán al conjunto de prueba.
#random_state=42 fija la semilla para que los resultados sean reproducibles.


'''
Crea un modelo de regresión logística y entrénalo
'''
#1. DEBEMOS ESCALAR LOS DATOS: 
#La regresión logística funciona mejor cuando los datos tienen un rango similar. 
from sklearn.preprocessing import StandardScaler # transforma los datos para que tengan una media de 0 y una desviación estándar de 1

scaler = StandardScaler() #scaler, que se utilizará para estandarizar los datos.
X_train = scaler.fit_transform(X_train) #Transformación de los datos de entrenamiento
X_test = scaler.transform(X_test) #transformación de los datos de prueba
#los datos de X_train y X_test(aprendiendo la media y la desviación estándar de cada columna) y 
#transforma los valores de X_train y X_test para que estén estandarizados (cada columna tendrá una media de 0 y una desviación estándar de 1).


#2.Crear la regresión logística
clf = linear_model.LogisticRegression()  #Crear modelo de regresion logistica
clf.fit(X_train, y_train)  # Entrenamos el modelo con los datos de entrenamiento



'''
Haz predicciones para X_test
'''
y_pred = clf.predict(X_test) #Vamos a predecir 
print(f"Predicciones del modelo: {y_pred}")

# Calcular probabilidades NO ES NECESARIO 
#probabilities = clf.predict_proba(X_test)
#print("Probabilidades de cada clase:")
#print(probabilities)


'''
y calcula la exactitud del modelo con metrics.accuracy_score() para X_train e y_train
'''
# Calcula la exactitud del modelo en el conjunto de prueba
accuracy = accuracy_score(y_test, y_pred)
print(f"Exactitud del modelo en el conjunto de prueba: {accuracy:.2f}, es decir se ajusta {accuracy*100:.2f}%")