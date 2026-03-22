import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# 1. Create dataset
X, y = make_classification(n_samples=500, n_features=10, random_state=42)

# 2. Split data (IMPORTANT)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Create model (simple neural network)
# model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=500)
model = MLPClassifier(hidden_layer_sizes=(200, 200), max_iter=1000)

# 4. Train model
model.fit(X_train, y_train)

# 5. Predictions
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# 6. Accuracy
train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)

print("Training Accuracy:", train_acc)
print("Testing Accuracy:", test_acc)