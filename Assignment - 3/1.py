import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split


df = pd.read_csv('C:/Users/Acer/Downloads/archive/USA_Housing.csv')

# a
X = df.drop(['price'], axis=1).values
y = df['price'].values

# b
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# c
kf = KFold(n_splits=5, shuffle=True, random_state=42)

#d
best_r2 = -np.inf
best_beta = None

for train_index, test_index in kf.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]

    beta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
    y_pred = X_test @ beta
    r2 = r2_score(y_test, y_pred)

    if r2 > best_r2:
        best_r2 = r2
        best_beta = beta

print("Best R-squared:", best_r2)
print("Best beta:", best_beta)

#e
X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

y_pred_final = X_test_final @ best_beta
r2_final = r2_score(y_test_final, y_pred_final)

print("Final R-squared:", r2_final)