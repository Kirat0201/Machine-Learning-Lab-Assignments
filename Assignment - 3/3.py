import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

column_names = [
    "symboling", "normalized_losses", "make", "fuel_type", "aspiration", "num_doors", "body_style",
    "drive_wheels", "engine_location", "wheel_base", "length", "width", "height", "curb_weight",
    "engine_type", "num_cylinders", "engine_size", "fuel_system", "bore", "stroke",
    "compression_ratio", "horsepower", "peak_rpm", "city_mpg", "highway_mpg", "price"
]
df = pd.read_csv('imports-85.data', names=column_names, na_values='?')

for col in df.columns:
    if df[col].dtype == 'O':
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)
df.dropna(subset=['price'], inplace=True)
df['price'] = pd.to_numeric(df['price']) 

doors_map = {'two':2, 'four':4}
cyl_map = {'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'eight':8, 'twelve':12}
df['num_doors'] = df['num_doors'].map(doors_map).astype(int)
df['num_cylinders'] = df['num_cylinders'].map(cyl_map).astype(int)

df = pd.get_dummies(df, columns=['body_style', 'drive_wheels'])

for col in ['make', 'aspiration', 'engine_location', 'fuel_type']:
    df[col] = LabelEncoder().fit_transform(df[col])

df['fuel_system'] = df['fuel_system'].apply(lambda x: 1 if 'pfi' in x else 0)

df['engine_type'] = df['engine_type'].apply(lambda x: 1 if 'ohc' in x else 0)

for col in df.columns:
    if df[col].dtype == 'O':
        df[col] = pd.to_numeric(df[col], errors='coerce')

X = df.drop(['price'], axis=1).values
y = df['price'].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
r2_initial = r2_score(y_test, y_pred)
print("R2 score without PCA:", r2_initial)

pca = PCA(n_components=0.95) 
X_pca = pca.fit_transform(X_scaled)
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.3, random_state=42)
lr_pca = LinearRegression()
lr_pca.fit(X_train_pca, y_train_pca)
y_pred_pca = lr_pca.predict(X_test_pca)
r2_pca = r2_score(y_test_pca, y_pred_pca)
print("R2 score with PCA:", r2_pca)