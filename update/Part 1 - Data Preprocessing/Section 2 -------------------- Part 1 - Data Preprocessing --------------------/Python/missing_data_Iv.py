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

# REEMPLAZO NaN
# New in version 0.20: SimpleImputer replaces the previous sklearn.preprocessing.Imputer estimator which is now removed.
imputer = SimpleImputer(missing_values=np.NAN, strategy="mean")
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

