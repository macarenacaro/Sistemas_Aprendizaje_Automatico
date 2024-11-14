#**************************** INSPECCIÓN DEL DATASET ****************************
import pandas as pd
import numpy as np

#**************************** 1. Cargar Datos:****************************
df = pd.read_csv('adult.csv', header=0)
print("COLUMNAS:", df.columns) #Nombre de cada column

#**************************** 2. Mostrar datos:****************************
print(df['hours-per-week'].isnull().sum()) #Cuenta cuántos valores nulos (NaN) hay en la columna hours-per-week. #8

#**************************** 3. Rellenar nulos con 0:****************************
df['hours-per-week'].fillna(0, inplace=True)

#**************************** 4. Distribución de horas trabajadas:****************************
print("HORAS TRABAJADAS\n",df['hours-per-week'].value_counts()) #muestra cuántas veces aparece cada valor de horas trabajadas
print(df.info())

#**************************** 5. Ver cuantos tienen menos de 40 hrs:****************************
print("MENOS DE CUARENTA\n",(df['hours-per-week']<40).value_counts()) 


#**************************** 6.Crear categorías (bins) de horas trabajadas ****************************
#Esto permite clasificar los empleados según el tiempo que trabajan

df['Hour-bin'] = df['hours-per-week']
df.loc[df['Hour-bin'] == 0, 'Hour-bin'] = 0  # No trabaja
df.loc[df['Hour-bin'] < 20, 'Hour-bin'] = 1  # Menos de media jornada
df.loc[(df['Hour-bin'] >= 20) & (df['Hour-bin'] < 40), 'Hour-bin'] = 2  # Media jornada
df.loc[(df['Hour-bin'] == 40), 'Hour-bin'] = 3  # Jornada completa
df.loc[(df['Hour-bin'] > 40), 'Hour-bin'] = 4  # Superior a jornada completa

#**************************** 7.Crear columnas dummies ****************************
#Dummies también son categoricos
df['No trabaja'] = df['Hour-bin'].map(lambda s: 1 if s == 0 else 0)
df['Menos de media jornada'] = df['Hour-bin'].map(lambda s: 1 if s == 1 else 0)
df['Media jornada'] = df['Hour-bin'].map(lambda s: 1 if s == 2 else 0)
df['Jornada completa'] = df['Hour-bin'].map(lambda s: 1 if s == 3 else 0)
df['Superior a jornada completa'] = df['Hour-bin'].map(lambda s: 1 if s == 4 else 0)

print(df.head())