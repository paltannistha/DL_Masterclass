import numpy as np

# Original labels
y = np.array([0, 1, 2])

# Number of classes
num_classes = 3

# One-hot encoding
y_one_hot = np.eye(num_classes)[y]

print("Original y:", y)
print("One-hot encoded y:")
print(y_one_hot)