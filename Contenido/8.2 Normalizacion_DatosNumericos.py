#**************************** INSPECCIÓN DEL DATASET ****************************
#******************* TRATAMIENTO DE NaN ****************************

import pandas as pd
import numpy as np

#**************************** 1. Cargar Datos:****************************
df = pd.read_csv('adult.csv', header=0)
print("COLUMNAS:", df.columns) #Nombre de cada column

#**************************** 2. Mostrar datos:****************************
print("DATA-SHAPE: ",df.shape) #Cuantos datos existen (nº filas, nºcolumn)
print(df.head(10)) #muestra 10 filas

#**************************** 3.VALORES NULOS:****************************
print("VALORES NULOS:\n", df.isnull().sum()) #Cuantos valores nulos hay en cada columna

#**************************** 4. Valores nulos en una columna particular****************************
print(df[df['education-num'].isna()]) # Veamos los valores NaN del feature education-num

#**************************** 5. Fusionar columnas Co-relacionadas****************************
group = df[['education','education-num']].groupby('education').mean().reset_index()

#df[['education', 'education-num']]: Selecciona las columnas education y education-num
#.groupby('education'): Agrupa los datos por la columna education, lo que significa que los datos se agruparán según cada nivel educativo.
#.mean(): Calcula el promedio de la columna education-num para cada grupo de education.
#.reset_index(): Convierte el resultado de la operación de agrupación en un nuevo DataFrame

print("NUEVO GRUPO:", group)


#**************************** 6. Unir nuevo grupo con Df****************************
df = df.merge(group, how='left', on='education',)
print(df.isnull().sum())# ver cuantos nulos tiene lo que hemos unido
"""education                   0
education-num_x             6
education-num_y             0""" #tenemos 3 education, debemos mantener la y, xq tiene 0 nulos y es de numeros


#**************************** 7. Unir nuevo grupo con Df****************************
df.drop('education-num_x', axis=1, inplace=True) #eliminar education num 


#*************************** 8. Renombrar col ****************************
df.rename(columns={'education-num_y': 'education-num'}, inplace=True)

#************************** 9. Eliminar education ****************************
df.drop('education', axis=1, inplace=True)
