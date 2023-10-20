# imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# REEMPLAZO NaN
# New in version 0.20: SimpleImputer replaces the previous sklearn.preprocessing.Imputer estimator which is now removed.
df_data = pd.read_csv("update/Part 1 - Data Preprocessing/Section 2 -------------------- "
                      "Part 1 - Data Preprocessing --------------------/Python/Data.csv")
X = df_data.iloc[:, :-1].values
y = df_data.iloc[:, -1].values
imputer = SimpleImputer(missing_values=np.NAN, strategy="mean")
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Variables categoricas ordinales (importa el orden como por ejemplo las tallas de ropa)
le_X = LabelEncoder()
X[:, 0] = le_X.fit_transform(X[:, 0])

# Variables categoricas DUMMY (no importa el orden)
# old
# onehotencoder = OneHotEncoder(categories=[0])
# X = onehotencoder.fit_transform(X).toarray()
# new
ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],
                       remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=float)

# Ejemplo con la variable dependiente (solo toma 2 valores, lo podemos hacer con un label)
le_y = LabelEncoder()
y = le_y.fit_transform(y)

