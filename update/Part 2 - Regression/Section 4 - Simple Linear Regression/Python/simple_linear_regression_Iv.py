# Simple Linear Regression

# Importing the libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

# Importing the dataset
os.getcwd()
try:
    os.chdir("update/Part 2 - Regression/Section 4 - Simple Linear Regression/Python")
except Exception as e:
    print(e)
dataset = pd.read_csv('Salary_Data.csv')
dataset.head()
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# # Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
# X_test[:, 3:] = sc.transform(X_test[:, 3:])
# print(X_train)
# print(X_test)

# Creacion del modelo segun los valores de entrenamiento
# No hace falta escalar los datos, la propia libreria de python para regresion hace el trabajo sucio
regression = LinearRegression()
regression.fit(X_train, y_train)

# Predecir el conjunto de test
y_pred = regression.predict(X_test)

# Visualizar los resultados del entrenamiento
plt.scatter(X_train, y_train, color="red")
plt.plot(X_train, regression.predict(X_train), color="blue")
plt.title("Sueldo VS Años de Experiencia (Conjunto de entrenamiento)")
plt.xlabel("Años de experiencia")
plt.ylabel("Sueldo (en $)")
plt.show()

# Visualizar los resultados del test
plt.scatter(X_test, y_test, color="red")
plt.plot(X_train, regression.predict(X_train), color="blue") # la recta de regresion siempre es la misma
plt.title("Sueldo VS Años de Experiencia (Conjunto de TEST)")
plt.xlabel("Años de experiencia")
plt.ylabel("Sueldo (en $)")
plt.show()