# 🧠 One-Hot Encoding (Deep Learning Basics)

## 📌 What is it?

One-hot encoding is a way to convert class labels into binary vectors so that a machine learning model can understand them.

---

## ❓ Why is it needed?

Machine learning models work with numbers.

If we use:
- 0 = Cat
- 1 = Dog
- 2 = Horse


The model may wrongly assume there is an order (2 > 1 > 0).

So we remove this confusion using one-hot encoding.

---

## 🔄 How it works

Each class is converted into a vector:

- Cat  → [1, 0, 0]  
- Dog  → [0, 1, 0]  
- Horse → [0, 0, 1]

---

## 🧪 Example

Original labels:

The model may wrongly assume there is an order (2 > 1 > 0).

So we remove this confusion using one-hot encoding.

---

## 🔄 How it works

Each class is converted into a vector:

- Cat  → [1, 0, 0]  
- Dog  → [0, 1, 0]  
- Horse → [0, 0, 1]

---

## 🧪 Example

Original labels:
y = [0, 1, 2, 1, 0]

One-hot encoded:

[
[1, 0, 0],
[0, 1, 0],
[0, 0, 1],
[0, 1, 0],
[1, 0, 0]
]

---

## ⚙️ Python Example

```python
import numpy as np

y = np.array([0, 1, 2, 1, 0])
num_classes = 3

y_one_hot = np.eye(num_classes)[y]
print(y_one_hot)
