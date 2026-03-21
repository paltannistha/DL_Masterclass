import random

# Training data
X = [1, 2, 3, 4, 5]
Y = [2, 4, 6, 8, 10]

# Initialize weight and bias randomly
w = random.random()
b = random.random()

# Learning rate
lr = 0.01

# ReLU activation
def relu(x):
    return max(0, x)

# Training loop
for epoch in range(100):

    total_loss = 0

    for x, y_true in zip(X, Y):
        
        # Forward pass
        y_pred = relu(w * x + b)

        # Loss (MSE)
        loss = (y_true - y_pred) ** 2
        total_loss += loss

        # Simple gradient (manual intuition)
        # derivative of loss wrt prediction = -2*(y_true - y_pred)
        grad = -2 * (y_true - y_pred) # it turns the error into a signal for correction

        # Update weights (very simplified)
        w = w - lr * grad * x
        b = b - lr * grad

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

# Test
print("\nTesting:")
for x in X:
    print(f"Input: {x}, Prediction: {relu(w*x + b):.2f}")