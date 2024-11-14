#**************************** INSPECCIÓN DEL DATASET ****************************
import pandas as pd
import numpy as np

#**************************** 1. Cargar Datos:****************************
df = pd.read_csv('adult.csv', header=0)
print("COLUMNAS:", df.columns) #Nombre de cada column

#**************************** 2. Mostrar datos:****************************
print("DATA-SHAPE: ",df.shape) #Cuantos datos existen (nº filas, nºcolumn)
print(df.head(10)) #muestra 10 filas

print("INFORMACIÓN",df.info()) # Muestra informacion de datos
# por ej columna AGE:   age  32559 non-null (le falta 1 valor para 32560)  float64


print("DESCRIBE:",df.describe()) #df.describe(include='all')
#count: Indica cuántos valores existen en cada columna
#mean: Es el promedio de los valores en cada columna.
#std: Es la desviación estándar, una medida de cuánto se dispersan los valores alrededor del promedio. Un valor más alto significa que los datos están más dispersos. 
# Por ejemplo, capital-gain tiene una desviación estándar alta (7385.4), lo que sugiere gran variación en los valores de esta columna.
#min: Es el valor mínimo encontrado en cada columna. En age, el valor mínimo es 17.
#25%: Es el primer cuartil, es decir, el valor en el que el 25% de los datos son menores. En age, el 25% de las personas tienen 28 años o menos.
#50%: Es la mediana o el segundo cuartil, el valor que está justo en el medio de todos los datos. En age, la mitad de las personas tienen 37 años o menos.
#75%: Es el tercer cuartil, donde el 75% de los valores son menores. En age, el 75% de las personas tienen 48 años o menos.
#max: Es el valor máximo encontrado en cada columna. Para capital-gain, el valor máximo es 99,999, lo que indica un valor de ganancia de capital excepcionalmente alto.


#**************************** PREPROCESANDO DATOS ****************************


#*******************3. SUSTITUCIÓN [sex] ****************************

#***** 3.1 Datos categóricos ****** : 
#Tipo de datos que se pueden almacenar en grupos o categorías con nombres o etiquetas

#***** 3.1.1 Modificar datos categóricos ****** : 
print(df['sex'].unique()) # Mostramos los valores que puede tomar:

# Vamos a crear una nueva columna
df['Sex'] = df['sex'].replace({' Male': 1, ' Female':0}) #Debemos fijarnos que estén exactemente escritos
print(df.head(10)) #muestra 10 filas
df.drop(['sex'], axis=1, inplace=True) #Eliminar la Antigua




#**************************** 4.LABEL ENCODER ****************************
#******************* PROCESAR DATOS CATEGORICOS A ENTEROS ****************************

#*****4.1 librería scikit-learn que se usa en machine learning para convertir datos categóricos (es decir, datos que representan categorías o clases) en números enteros
print("WORKCLASS UNIQUE",df['workclass'].unique()) #Mostramos los valores que puede tomar de la col Workclass

print("WORKCLASS SHAPE",df['workclass'].shape) #(32560,1) tiene el total de datos esperados

print("WORKCLASS COUNTS:", df['workclass'].value_counts())
#Vemos que aparece un valor ? además 3 valores vacios.

#*****4.2 Vamos a eliminar los valores vacios y los valores que no se encuentran en la lista de categorías
df.drop(df[df['workclass'] == ' '].index, inplace = True) #Los escribimos tal como aparecen en WORKCLASS
df.drop(df[df['workclass'] == ' ?'].index, inplace = True)

#Ya no existen
print("WORKCLASS COUNTS2:", df['workclass'].value_counts())
print("WORKCLASS SHAPE2",df['workclass'].shape)
print("WORKCLASS UNIQUE2",df['workclass'].unique())

#*****4.3 CONVERTIR LAS CATEGORÍAS A NUMEROS ****************
from sklearn import preprocessing
encoder = preprocessing.LabelEncoder() #Crea un objeto LabelEncoder que se usará para convertir las categorías en números.
encoded = encoder.fit_transform(df['workclass']) #encoded ahora contiene una matriz de números que representan las categorías originales
df['workclass'] = encoded.astype('int') #Reemplaza workclass con los valores codificados (numéricos).


#*****4.4 Obtener las categorías originales, solo las muestra, ya están hechas numeros (NO LE VEO SENTIDO)
encoder.inverse_transform(df.workclass) #convertir los números codificados de vuelta a sus categorías originales

#***** 4.5 Ver las categorías originales en orden 
print("WORKCLASS ENCONDER CLASES", encoder.classes_)

#***** 4.6 Ver la distribución de las clases 
print ("WORKCLASS CLASES COUNTS\n",df['workclass'].value_counts())
#3    22693  ' Private'
#5     2541  ' Self-emp-not-inc'
#1     2093 ' Local-gov' 
#6     1297 ' State-gov' .....



#**************************** 5.ONE HOT ****************************
#******************* PROCESAR DATOS CATEGORICOS A ENTEROS ****************************

#*****5.1 Mostrar y contar valores
print("RACE COUNTS", df['race'].value_counts()) #cuenta cuántas veces aparece cada categoría en la columna race
print("RACE UNIQUE",df['race'].unique())#muestra todas las categorías únicas en la columna race, incluyendo un valor vacío (' ')

#*****5.3   Eliminamos las filas en los que race = ' '
df.drop(df[df['race'] == ' '].index, inplace = True) #Busca filas donde race es ' '  y elimina esas filas
print("RACE COUNTS2",df['race'].value_counts())

#*****5.4  Crear Columnas en numeros 
dummies = pd.get_dummies(df['race'], prefix='race') # toma cada valor único en race y crea una columna nueva para cada uno.
print ("RACE DUMMIES \n",dummies) #el término "dummies" en realidad solo describe las columnas binarias resultantes, en pandas, no sognifica que estoy utilizando metodo dummies

#*****5.5  Agregar a tabla original y eliminar la columna race
df = pd.concat([df, dummies], axis=1) # agrega las nuevas columnas dummy al DataFrame df.
df.drop(['race'], axis=1, inplace=True) #elimina la columna original race para evitar duplicados.
print(df)

#**************************** ¡¡¡5.ONE HOT VERSION MACA!!! ****************************
#Hice este porque aqui se importa OneHotEncoder, no se usa dummies, es decir, ambos hacen lo mismo pero son metodos diferentes
#en el ejemplo anterior se explicó oneHot con dummies, no le encontraba la diferencia. 

from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# Crear un ejemplo de DataFrame similar al tuyo
data = {'race': ['White', 'Black', 'Asian', 'White', 'Black', 'White']}
df = pd.DataFrame(data)

# **5.1 Mostrar y contar valores**
print("RACE COUNTS", df['race'].value_counts())  # Cuenta cuántas veces aparece cada categoría en la columna 'race'

# **5.3 Eliminar filas con valores no deseados** (aunque en este ejemplo no tenemos espacios vacíos, lo harías como en tu código)
df = df[df['race'] != ' ']  # Esto eliminaría filas con valores vacíos

# **5.4 OneHotEncoder de scikit-learn**
encoder = OneHotEncoder(sparse=False)  # Sparse=False devuelve una matriz densa (no dispersa), si prefieres matrices dispersas puedes poner sparse=True

# Transformar los datos categóricos (aplica One-Hot Encoding)
encoded_race = encoder.fit_transform(df[['race']])  # El parámetro fit_transform hace el ajuste y la transformación en un solo paso

# Convertir el resultado en un DataFrame para un manejo más fácil
encoded_race_df = pd.DataFrame(encoded_race, columns=encoder.get_feature_names_out(['race']))

# **5.5 Agregar las nuevas columnas codificadas a la tabla original**
df = pd.concat([df, encoded_race_df], axis=1)
df.drop(['race'], axis=1, inplace=True)  # Eliminar la columna original para evitar duplicados

# Mostrar el DataFrame final
print(df)




#**************************** 6. Encode_Label ****************************
#*****6.1 Crear Funcion para crear la columna de manera numerica
def encode_label(df):
    return df.astype('category').cat.codes
#convierte la columna de texto a un tipo category, que es una forma especial de almacenar texto en pandas
#cat.codes asigna un número único a cada categoría en la columna.
# Nota: No podemos decodificar como con LabelEncoder


#*****6.2 Mostrar y contar valores de columna relationship
print("RELATIONSHIP COUNTS\n" ,df['relationship'].value_counts()) #cuántas veces aparece cada categoría en la columna

#*****6.3 CONVERTIR LAS CATEGORÍAS A NUMEROS 
df['relationship'] = encode_label(df['relationship']) #función encode_label para transformar la columna relationship de texto a números.

#*****6.4 Ver Relationship modificada a número 
print(df)


#*****6.5 BONUS: SE PUEDE APLICAR LO MISMO A VARIAS
df['occupation'] = encode_label(df['occupation'])
df['native-country'] = encode_label(df['native-country'])
df['gains'] = encode_label(df['gains'])





"""EL PROBLEMA DE ESTO, ES QUE NO PODEMOS VER A QUE ELEMENTO CORRESPONDE EL Nº 0, 1 ,2...  el proximo caso lo haremos de 
manera diferente para poder apreciar eso (VERSION MACA)"""
#**************************** 7. Enconde_Label Maca ****************************
#***** 7.0: Definir la función para codificar etiquetas
def encode_label(df): 
    return df.astype('category').cat.codes

#***** 7.1: Ver las categorías antes de codificar
print("Categorías originales en 'marital-status':", df['marital-status'].unique())


#***** 7.2 Convierte la columna  a tipo category 
df['marital-status'] = df['marital-status'].astype('category')  #para poder visualizar el orden en que las categorías serán codificadas.

#***** 7.3 Ver las categorías ordenadas y asignación de códigos
# Ver categorías ordenadas en marital-status , es como un COUNT
print("Categorías ordenadas en 'marital-status':", df['marital-status'].cat.categories)

#***** 7.3 Ver los códigos únicos asignados y cómo están en el DataFrame
print("Códigos asignados en 'marital-status':")
print(df['marital-status'].cat.codes.unique()) #[2 0 3 4 5 1 6]  Divorced, Married....

#***** 7.4: Aplicar la codificación
df['marital-status'] = encode_label(df['marital-status'])

#***** 7.5 Mostrar los primeros valores codificados
print("Valores codificados en 'marital-status':")
print(df[['marital-status']].head())
