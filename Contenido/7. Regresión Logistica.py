import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

# Load data
# data = np.loadtxt('./heights_weights.csv', delimiter=',', skiprows=1)

#**************************** 1. Cargar Datos:****************************

df = pd.read_csv('https://drive.google.com/uc?export=download&id=1ypiN0Kalkwsu6ygDomGRNVi5x2Fj-8TO')

#**************************** 2. Mostrar datos****************************

print("Data shape: ",df.shape)
print(df.head(10))


#**************************** 3. Seleccionar una Xs e Ys ****************************

X = df.values[:,1:3] # Altura y Peso
y = df.values[:,0] # La fila de Género (los valores 0 para Female, valores 1 para Male)  ¿como lo sé, no se?

#SI NUESTRO "Y" SERÁ GÉNERO, DEBEMOS TENER 2 PROBABILIDADES PARA LA CATEGORÍA (MUJER,HOMBRE)
categories = np.array(['Female', 'Male']) #Asignamos los valores de genero como categorias
print("X shape:", X.shape)#X shape: (10000, 2) = (numero de datos, 2 columnas)


#**************************** 4. Entrenamiento ****************************

clf = linear_model.LogisticRegression() #Crear un modelo de regresión logística 
clf.fit(X, y) #ajusta el modelo a nuestros datos x e y, aqui hace funcion sigmoidal :D

#**************************** 5. Hacer Predicciones de Categorías ****************************

prediction_result = clf.predict([[70,180]]) #tenemos una persona que mide 70 pulgadas y pesa 180 libras
print(f"La predicción de categoría es: {categories[int(prediction_result)]} , para height: 70, weight: 180:")

#**************************** 6. Coeficientes, Intercepto y Resultado de la Predicción ****************************
#Los coeficientes y el intercepto son los valores que el modelo ha aprendido para tomar decisiones.

print("Coefs:", clf.coef_)
print("Intercept:", clf.intercept_)
print("resultados de la predicción:",prediction_result)

#**************************** 7. Graficar la Frontera de Decisión ****************************
#Mostrar todos nuestros y separarlos por la categoria que le hemos dado
# Para trazar la frontera de decisión
parameters = clf.coef_ #extrayendo los coeficientes (pesos) que el modelo ha aprendido
x_values = [np.min(X[:, 0]), np.max(X[:, 0])] #generando valores de altura para el rango de altura
y_values = - (parameters[0] + np.dot(X[1], x_values)) / parameters[0] #Calcular la frontera de decisión
#Aquí se calcula el valor de la variable dependiente (peso) para cada valor de altura.

#***************** 7.1 Intercepto y Coef de la Frontera de Desición 
b = clf.intercept_[0]
w1, w2 = clf.coef_.T

# Calcular el intercepto y la pendiente de la frontera de decisión.
c = -b/w2
m = -w1/w2
#Estos valores son cruciales para dibujar la línea que representa la frontera de decisión.


#***************** 7.2 Graficar la Frontera de Decisión 

#***** 7.2.1 Definimos los límites del gráfico (mínimo y máximo para altura y peso), 
xmin, xmax = min(X[:,0]), max(X[:,0])
ymin, ymax = min(X[:,1]), max(X[:,1])

#***** 7.2.2 Graficar los puntos de datos
xd = np.array([xmin, xmax])
yd = m*xd + c # Usando la fórmula de la recta
plt.plot(xd, yd, 'k', lw=1, ls='--')  # Graficar la línea

#***** 7.2.3 Rellenar el área encima y debajo de la línea
plt.fill_between(xd, yd, ymin, color='tab:orange', alpha=0.2)  # Área para una clase
plt.fill_between(xd, yd, ymax, color='tab:blue', alpha=0.2) # Área para otra clase
plt.legend()

#***** 7.2.3 Graficar los Datos de Hombres y Mujeres
X_male = df.loc[df['Gender'] == 1.0]
X_female = df.loc[df['Gender'] == 0.0]
print(X_male.head(5))
print(X_female.head(5))
plt.scatter(X_male.iloc[:, 1], X_male.iloc[:, 2], s=10, label='Male', alpha=0.2)
plt.scatter(X_female.iloc[:, 1], X_female.iloc[:, 2], s=10, label='Female', alpha=0.2)

plt.legend()
plt.show()



#**************************** 8. Evaluación del modelo ****************************
#***************** 8.1 Coeficient R2: 
# Nuestro modelo se ajusta % a los datos 

from sklearn import metrics
y_pred = clf.predict(X)
print(f"Nuestro modelo se ajusta {metrics.r2_score(y, y_pred)}  % a los datos")

#*****************8.2 Metrica de Exactitud: 
# En la Regresión Logística es mejor utilizar como métrica la exactitud (Porcentaje de aciertos):

print(f"La metrica de exactitud se ajusta: {metrics.accuracy_score(y, y_pred)}  % a los datos")



#**************************** 9. Ejemplo de Predicción ****************************
height = 68
weight = 172

#***************** 9.1. OPCION 1: Prediccion de Género
y_pred_1 = clf.predict([[height, weight]])
print(f"Resultado de y_pred_1: {y_pred_1}") #[1.] #Segun categoria es Male 


#***************** 9.2. OPCION 1: Predicción de género con f(x) Sigmoide
import math
y_pred_calc = height*clf.coef_[0][0] + weight*clf.coef_[0][1] + clf.intercept_ #Ecuacion de regresion lineal
y_pred_calc_sigm = 1 / (1 + math.exp(-y_pred_calc)) # 1/1+e^-(ecuacion de regresion lineal)
print(f"Resultado de y_pred_calc_sigm: {y_pred_calc_sigm}, es sobre 0,5, por ende es MALE")
