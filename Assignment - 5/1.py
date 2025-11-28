from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
import random

digits = load_digits()
images = digits.images
data = digits.data
target = digits.target
samples = images.shape[0]
flat_images = images.reshape(samples, -1)
idx = random.sample(range(samples), 6)
fig, axes = plt.subplots(2, 3, figsize=(8, 6))
for ax, i in zip(axes.flatten(), idx):
    ax.imshow(images[i], cmap='gray')
    ax.set_title(str(target[i]))
    ax.axis("off")
plt.tight_layout()
plt.show()

x_train, x_test, y_train, y_test = train_test_split(flat_images, target, test_size=0.2, random_state=42)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model = LogisticRegression(multi_class='ovr', random_state=42)
model.fit(x_train_scaled, y_train)
training_accuracy = model.score(x_train_scaled, y_train)
test_accuracy = model.score(x_test_scaled, y_test)
print(f"Training accuracy: {training_accuracy:.2f}")
print(f"Test accuracy: {test_accuracy:.2f}")

idx = 2
probability = model.predict_proba([x_test_scaled[idx]])[0]
print("Probabilities:", probability)
predicted_class = np.argmax(probability)
print("Predicted class:", predicted_class)
print("Actual class:", y_test[idx])

cm = confusion_matrix(y_test, model.predict(x_test_scaled))
display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
display.plot(cmap="Greens")
plt.title("Confusion Matrix")
plt.show()

cm_copy = cm.copy()
np.fill_diagonal(cm_copy, 0)
confused_indices = np.dstack(np.unravel_index(np.argsort(cm_copy.ravel())[::-1], cm_copy.shape))[0]
print("\nTop 3 most confused digit pairs (true → predicted):")
for i in range(3):
    true_label, pred_label = confused_indices[i]
    print(f"{true_label} → {pred_label} ({cm_copy[true_label, pred_label]} times)")

C_values = [0.01, 0.1, 1, 10]
test_accuracies = []
for C in C_values:
    clf = LogisticRegression(multi_class='ovr', random_state=42, max_iter=1000, C=C)
    clf.fit(x_train_scaled, y_train)
    acc = clf.score(x_test_scaled, y_test)
    test_accuracies.append(acc)
plt.plot(C_values, test_accuracies, marker='o')
plt.xscale('log')
plt.xlabel("C (inverse regularization strength)")
plt.ylabel("Test Accuracy")
plt.title("Effect of Regularization (C) on Test Accuracy")
plt.grid(True)
plt.show()
print("\nTest accuracies for different C values:")
for C, acc in zip(C_values, test_accuracies):
    print(f"C={C}: {acc:.4f}")

models = {
    "Default (C=1, L2)": LogisticRegression(multi_class='ovr', random_state=42, max_iter=1000, C=1),
    "Weak Reg (C=1e6)": LogisticRegression(multi_class='ovr', random_state=42, max_iter=1000, C=1e6),
    "No Reg (penalty=None)": LogisticRegression(multi_class='ovr', random_state=42, max_iter=1000, penalty=None)
}
print("\nTest accuracies with different regularization:")
for name, m in models.items():
    m.fit(x_train_scaled, y_train)
    acc = m.score(x_test_scaled, y_test)
    print(f"{name}: {acc:.4f}")

W = model.coef_ 
b = model.intercept_ 
x = x_test_scaled[idx]
z = np.dot(W, x) + b
p = 1 / (1 + np.exp(-z))
p_norm = p / np.sum(p)
print("\nManual OvR probability computation:")
print("Unnormalized (sigmoid) probabilities:", np.round(p, 3))
print("Normalized probabilities:", np.round(p_norm, 3))
print("Sklearn predict_proba:", np.round(probability, 3))