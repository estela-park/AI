import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.backend import sigmoid


# Data: below is a Xor-gate: when parts disagree it's whole True
x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_data = [0, 1, 1, 0]


# Modelling
model = Sequential()
model.add(Dense(64, input_dim=2, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, input_dim=2, activation='sigmoid'))


# Training
model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(x_data, y_data, batch_size=1, epochs=240)


# Evaluation & Prediction
y_predict = model.predict(x_data)
result = model.evaluate(x_data, y_data)
# acc = accuracy_score(y_data, y_predict)

print(f'prediction: {y_predict}, acc: {result}')

'''
For Multi-layer Perceptron, batch_size=1, epochs=120
    prediction: [[0.01024322] [0.9898919 ] [0.9869681 ] [0.01659427]], acc: 0.012576669454574585
 > epochs=240
    prediction: [[8.6763495e-04] [9.9886107e-01] [5.6324595e-01] [2.6251636e-03]], acc: 0.1446687877178192
'''