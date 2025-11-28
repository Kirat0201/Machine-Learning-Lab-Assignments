import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)

X1 = np.random.randn(1000)
X2 = 0.9 * X1 + 0.1 * np.random.randn(1000)
X3 = 0.8 * X1 + 0.2 * np.random.randn(1000)
X4 = 0.85 * X1 + 0.15 * np.random.randn(1000)
X5 = 0.88 * X1 + 0.12 * np.random.randn(1000)
X6 = 0.9 * X1 + 0.05 * np.random.randn(1000)
X7 = 0.75 * X1 + 0.25 * np.random.randn(1000)
X = np.column_stack([X1, X2, X3, X4, X5, X6, X7])
y = 5 * X1 + 3 * X2 + 2 * X3 + np.random.randn(1000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)
X_train_norm = (X_train - X_mean) / X_std
X_test_norm = (X_test - X_mean) / X_std

def ridge_regression_gradient_descent(X, y, learning_rate, reg_lambda, num_iterations=1000):
    m, n = X.shape
    theta = np.zeros(n)
    for i in range(num_iterations):
        y_pred = X.dot(theta)
        gradient = -(2/m) * X.T.dot(y - y_pred) + 2 * reg_lambda * theta
        theta = theta - learning_rate * gradient
    return theta

learning_rates = [0.0001, 0.001, 0.01, 0.1, 1, 10]
regularization_params = [1e-15, 1e-10, 1e-5, 1e-3, 0, 1, 10, 20]

best_r2 = -float('inf')
best_params = None
best_theta = None

for lr in learning_rates:
    for reg_lambda in regularization_params:
        theta = ridge_regression_gradient_descent(X_train_norm, y_train, lr, reg_lambda)
        y_pred = X_test_norm.dot(theta)
        if np.any(np.isnan(y_pred)):
            print(f"NaN values encountered for lr={lr}, reg_lambda={reg_lambda}")
            continue
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        if r2 > best_r2:
            best_r2 = r2
            best_params = (lr, reg_lambda)
            best_theta = theta

print("Best Learning Rate:", best_params[0])
print("Best Regularization Parameter:", best_params[1])
print("Best R2 Score:", best_r2)