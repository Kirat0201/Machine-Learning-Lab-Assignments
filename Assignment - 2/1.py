import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from numpy import dot
from numpy.linalg import norm
# (I)
file = pd.read_csv('C:/Users/Acer/Downloads/archive/AWCustomers.csv')
selected_features = ['CustomerID','BirthDate','Education','Occupation','Gender','MaritalStatus','HomeOwnerFlag','NumberCarsOwned','TotalChildren','YearlyIncome']
df = file[selected_features]
print(df)
# (II)
# (a)
df = df.dropna()
print(df)
# (b)
scaler = MinMaxScaler()
df[['YearlyIncome']] = scaler.fit_transform(df[['YearlyIncome']])
print(df)
# (c)
df['Income Bin'] = pd.cut(df['YearlyIncome'], bins = 5, labels = False)
print(df);
# (d)
Scaler = StandardScaler()
df[['YearlyIncome']] = Scaler.fit_transform(df[['YearlyIncome']])
print(df)
# (e)
df_transformed = pd.get_dummies(df,columns = ['Gender','MaritalStatus','Education','Occupation','HomeOwnerFlag'],drop_first = True)
print(df_transformed)
# (III)
# (a)
def simple_matching(row1, row2):
    return sum(row1 == row2) / len(row1)
def jaccard_similarity(row1, row2):
    intersection = set(row1) & set(row2)
    union = set(row1) | set(row2)
    return len(intersection) / len(union)
def cosine_similarity(row1, row2):
    return dot(row1, row2) / (norm(row1) * norm(row2))
row1 = df_transformed.iloc[0]
row2 = df_transformed.iloc[1]
binary_columns = [col for col in df_transformed.columns if df_transformed[col].nunique() == 2]
jaccard = jaccard_similarity(row1[binary_columns], row2[binary_columns])
simple = simple_matching(row1, row2)
cosine = cosine_similarity(row1, row2)
print("Jaccard Similarity:", jaccard)
print("Simple Matching:", simple)
print("Cosine Similarity:", cosine)
# (b)
commute = df_transformed['NumberCarsOwned']
income = df_transformed['YearlyIncome']
correlation = commute.corr(income)
print(correlation)