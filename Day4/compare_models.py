import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Create dataset
X = np.linspace(-1, 1, 100)
y = X**3 + 0.1 * np.random.randn(100)

X = torch.tensor(X, dtype=torch.float32).view(-1, 1)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Split
X_train, X_test = X[:80], X[80:]
y_train, y_test = y[:80], y[80:]

# UNDERFITTING MODEL
underfit_model = nn.Sequential(
    nn.Linear(1, 1)
)

# OVERFITTING MODEL
overfit_model = nn.Sequential(
    nn.Linear(1, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 1)
)

# BALANCED MODEL
balanced_model = nn.Sequential(
    nn.Linear(1, 64),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

# Training function
def train(model, X, y, epochs=200):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for _ in range(epochs):
        preds = model(X)
        loss = loss_fn(preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Train models
train(underfit_model, X_train, y_train, 100)
train(overfit_model, X_train, y_train, 500)
train(balanced_model, X_train, y_train, 200)

# Predictions
with torch.no_grad():
    y_under = underfit_model(X)
    y_over = overfit_model(X)
    y_bal = balanced_model(X)

# Plot
plt.scatter(X.numpy(), y.numpy(), label="Data")
plt.plot(X.numpy(), y_under.numpy(), linestyle='dashed', label="Underfit")
plt.plot(X.numpy(), y_over.numpy(), label="Overfit")
plt.plot(X.numpy(), y_bal.numpy(), label="Balanced")
plt.legend()
plt.show()