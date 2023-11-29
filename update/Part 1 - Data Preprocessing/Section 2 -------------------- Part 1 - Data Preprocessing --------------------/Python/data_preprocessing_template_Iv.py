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

# dataset de entrenamiento y testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

# # Escalado de datos - No se hará siempre, por eso está comentado
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)
