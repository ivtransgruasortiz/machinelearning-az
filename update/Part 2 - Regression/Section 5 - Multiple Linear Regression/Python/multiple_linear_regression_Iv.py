# Simple Linear Regression

# Importing the libraries
import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

matplotlib.use('TkAgg', force=True)
print("Switched to:", matplotlib.get_backend())
# Importing the dataset
os.getcwd()
try:
    os.chdir("update/Part 2 - Regression/Section 5 - Multiple Linear Regression/Python")
except Exception as e:
    print(e)

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

labelencoder_X = LabelEncoder()
X[:, -1] = labelencoder_X.fit_transform(X[:, -1])
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [-1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Evitar la trampa de las variables ficticias
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Ajustar el modelo de regresion lineal multiple con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression

regresion = LinearRegression()
regresion.fit(X_train, y_train)

# Predicción de los resultados en el conjunto de test
y_pred = regresion.predict(X_test)

# Modelo optimo de regresion lineal multiple eliminando hacia atras variables
import statsmodels.formula.api as sm

X = np.append(arr=np.ones((len(X), 1)).astype(int), values=X, axis=1)
