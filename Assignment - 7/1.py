import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB

iris = load_iris()
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# step by step 
def gaussian_probability(x, mean, var):
    exp = 1e-9
    coeff = 1 / np.sqrt(2 * np.pi * var + exp)
    exponent  = np.exp(-((x - mean) ** 2) / (2 * var + exp))
    return coeff * exponent

classes = np.unique(y_train)
mean = {}
var = {}
priors = {}

for c in classes:
    x_c = x_train[y_train == c]
    mean[c] = x_c.mean(axis=0)
    var[c] = x_c.var(axis=0)
    priors[c] = x_c.shape[0] / x_train.shape[0]

def predict(X):
    y_pred = []
    for x in X:
        posteriors = []
        for c in classes:
            prior = np.log(priors[c])
            class_conditional = np.sum(np.log(gaussian_probability(x, mean[c], var[c])))
            posterior = prior + class_conditional
            posteriors.append(posterior)
        y_pred.append(classes[np.argmax(posteriors)])
    return np.array(y_pred)

y_pred = predict(x_test)

print("Step-by-Step Gaussian Naïve Bayes Classifier")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# in-built function
gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_pred = gnb.predict(x_test)
print("\nIn-built GaussianNB Classifier")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))