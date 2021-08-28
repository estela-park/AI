import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


# Data: below is a Xor-gate: when parts disagree it's whole True
x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_data = [0, 1, 1, 0]


# Modelling
model = LinearSVC()


# Training
model.fit(x_data, y_data)


# Evaluation & Prediction
y_predict = model.predict(x_data)
result = model.score(x_data, y_data)
acc = accuracy_score(y_data, y_predict)

print(f'prediction: {y_predict}, acc: {result} and {acc}')

# prediction: [1 1 1 1], acc: 0.5 and 0.5
# prediction: [1 0 1 1], acc: 0.25 and 0.25