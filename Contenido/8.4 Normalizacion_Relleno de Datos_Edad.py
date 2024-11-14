#**************************** RandomForestRegressor
# imputando valores predichos por machine learning
#estamos utilizando el método de aprendizaje automático RandomForestRegressor para llenar los valores 
# faltantes (NaN) en la columna age

#**************************** INSPECCIÓN DEL DATASET ****************************
import pandas as pd
import numpy as np

#**************************** 1. Cargar Datos:****************************
df = pd.read_csv('adult.csv', header=0)
print("COLUMNAS:", df.columns) #Nombre de cada column


#**************************** 2. Verificar si existen nulos:****************************
print("NULOS AGE:\n",df['age'].isna().value_counts())

#**************************** 3. Seleccionar columnas relevantes para predecir age****************************
df_sub = df[['age', 'education-num', 'hours-per-week', 'relationship']]

#**************************** 4. Crear conjunto entrenamiento X_train e Y_train****************************
X_train = df_sub.dropna(subset=['age']).drop('age', axis=1)
y_train = df['age'].dropna()

#X_train: Se eliminan los valores de la columna age para que solo contenga las características (education-num, hours-per-week, relationship).
#y_train: Contiene la columna age sin valores NaN, que serán las etiquetas (o valores objetivo) que el modelo intentará predecir.


#**************************** 5. Conjunto de prueba (X_test)****************************
X_test = df_sub.loc[np.isnan(df.age)].drop('age', axis=1)

#contiene las filas de df_sub donde age *es NaN.* 
# Estas son las observaciones para las cuales intentaremos predecir los valores de age usando el modelo de regresión.

#**************************** 6. Entrenamiento con RandomForestRegressor****************************
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300)
regressor.fit(X_train, y_train)

#**************************** 7. Predicción con el conjunto de prueba (X_test)****************************
y_pred = np.round(regressor.predict(X_test), 1)

#**************************** 8. Asignar los valores predichos en los lugares donde age estaba NaN ****************************
df.age.loc[df.age.isnull()] = y_pred

#***************************9. Verificar que ya no hayan valores Nan en Age ********************************
print(df['age'].isna().value_counts())


