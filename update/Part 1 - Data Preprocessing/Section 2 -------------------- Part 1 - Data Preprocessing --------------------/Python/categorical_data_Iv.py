# imports libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Cargar dataset y variables
df_data = pd.read_csv("update/Part 1 - Data Preprocessing/Section 2 -------------------- "
                      "Part 1 - Data Preprocessing --------------------/Python/Data.csv")
X = df_data.iloc[:, :-1].values
y = df_data.iloc[:, -1].values

# Codificar variables categoricas

# Variables categoricas ORDINALES (importa el orden como por ejemplo las tallas de ropa)
le_X = LabelEncoder()
X[:, 0] = le_X.fit_transform(X[:, 0])

# Variables categoricas DUMMY (no importa el orden)
ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],
                       remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=float)
# Para la variable dependiente y (solo toma 2 valores, lo podemos hacer con un label)
le_y = LabelEncoder()
y = le_y.fit_transform(y)

