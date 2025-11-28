import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

iris = load_iris()
X = iris.data  #
y = iris.target  
class_names = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(y, y_pred):
    m = len(y)
    cost = -(1/m) * np.sum(y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1 - y_pred + 1e-15))
    return cost

def gradient_descent(X, y, lr=0.01, num_iter=1000):
    m, n = X.shape
    weights = np.zeros(n)
    bias = 0
    for i in range(num_iter):
        z = np.dot(X, weights) + bias
        y_pred = sigmoid(z)
        dw = (1/m) * np.dot(X.T, (y_pred - y))
        db = (1/m) * np.sum(y_pred - y)
        weights -= lr * dw
        bias -= lr * db
        if i % 200 == 0:
            cost = compute_cost(y, y_pred)
            print(f"Iteration {i}, Cost: {cost:.4f}")
    return weights, bias

class OvRLogisticRegression:
    def __init__(self, lr=0.01, num_iter=1000):
        self.lr = lr
        self.num_iter = num_iter
        self.classes_ = None
        self.weights_ = {}
        self.biases_ = {}

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for c in self.classes_:
            print(f"\nTraining classifier for class {c} ({iris.target_names[c]}) vs Rest")
            y_binary = np.where(y == c, 1, 0)
            w, b = gradient_descent(X, y_binary, lr=self.lr, num_iter=self.num_iter)
            self.weights_[c] = w
            self.biases_[c] = b

    def predict_proba(self, X):
        """Compute probability for each class"""
        probs = np.zeros((X.shape[0], len(self.classes_)))
        for c in self.classes_:
            z = np.dot(X, self.weights_[c]) + self.biases_[c]
            probs[:, c] = sigmoid(z)
        return probs

    def predict(self, X):
        """Predict class with highest probability"""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

model = OvRLogisticRegression(lr=0.1, num_iter=2000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n✅ Model Evaluation:")
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", cm)