import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('C:/Users/Acer/Downloads/archive/USA_Housing.csv')

X = df.drop(['price'], axis=1).values
y = df['price'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_temp, X_test, y_temp, y_test = train_test_split(X_scaled, y, test_size=0.30, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

def gradient_descent(X, y, lr, n_iter):
    m, n = X.shape
    X_b = np.hstack([np.ones((m, 1)), X]) 
    beta = np.zeros(n + 1)
    for i in range(n_iter):
        y_pred = X_b @ beta
        error = y_pred - y
        grad = X_b.T @ error / m
        beta -= lr * grad
    return beta

lrs = [0.001, 0.01, 0.1, 1]
best_r2_val = -np.inf
best_beta = None
best_lr = None

for lr in lrs:
    beta = gradient_descent(X_train, y_train, lr, 1000)

    X_val_b = np.hstack([np.ones((X_val.shape[0], 1)), X_val])
    X_test_b = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
    
    y_pred_val = X_val_b @ beta
    y_pred_test = X_test_b @ beta
    r2_val = r2_score(y_val, y_pred_val)
    r2_test = r2_score(y_test, y_pred_test)
    
    print(f"Learning rate: {lr}")
    print(f"Validation R2: {r2_val:.4f}")
    print(f"Test R2: {r2_test:.4f}")
    print(f"Beta: {beta}\n")
    
    if r2_val > best_r2_val:
        best_r2_val = r2_val
        best_beta = beta
        best_lr = lr

print("Best validation R2:", best_r2_val)
print("Best learning rate:", best_lr)
print("Best beta:", best_beta)

y_pred_best_test = X_test_b @ best_beta
best_test_r2 = r2_score(y_test, y_pred_best_test)
print("Best Test R2 with best beta:", best_test_r2)
