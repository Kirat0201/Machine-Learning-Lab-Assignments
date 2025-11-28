import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("C:/Users/Acer/Downloads/Hitters.csv")

df = df.dropna(subset=["Salary"])
df = df.dropna()
categorical_cols = ["League", "Division", "NewLeague"]
numeric_cols = [c for c in df.columns if c not in categorical_cols + ["Salary"]]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(drop="first"), categorical_cols)
    ]
)

X = df.drop("Salary", axis=1)
y = df["Salary"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

lr_model = LinearRegression()
ridge_model = Ridge(alpha=0.5748)
lasso_model = Lasso(alpha=0.5748, max_iter=10000)
lr_model.fit(X_train_proc, y_train)
ridge_model.fit(X_train_proc, y_train)
lasso_model.fit(X_train_proc, y_train)

models = {
    "Linear Regression": lr_model,
    "Ridge (α=0.5748)": ridge_model,
    "Lasso (α=0.5748)": lasso_model
}

for name, model in models.items():
    y_pred = model.predict(X_test_proc)
    mse = mean_squared_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    print(f"{name} → MSE: {mse:.3f}, R2: {r2:.3f}")