import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, r2_score

boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.Series(boston.target, name='MEDV')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
ridge_alphas = np.logspace(-4, 4, 100)
ridge_cv = RidgeCV(alphas=ridge_alphas, store_cv_values=True)
ridge_cv.fit(X_train_scaled, y_train)
lasso_alphas = np.logspace(-4, 1, 100)
lasso_cv = LassoCV(alphas=lasso_alphas, cv=10, max_iter=10000, random_state=42)
lasso_cv.fit(X_train_scaled, y_train)

models = {
    "RidgeCV": ridge_cv,
    "LassoCV": lasso_cv
}

print("\nModel Evaluation Results:\n" + "-"*40)
for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name} Results:")
    print(f"  Best alpha: {model.alpha_:.4f}")
    print(f"  Test MSE: {mse:.4f}")
    print(f"  Test R²: {r2:.4f}\n")

print("Number of non-zero coefficients:")
print(f"RidgeCV: {(ridge_cv.coef_ != 0).sum()}")
print(f"LassoCV: {(lasso_cv.coef_ != 0).sum()}")