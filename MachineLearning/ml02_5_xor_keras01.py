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
model.add(Dense(1, input_dim=2, activation='sigmoid'))


# Training
model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(x_data, y_data, batch_size=1, epochs=120)


# Evaluation & Prediction
y_predict = model.predict(x_data)
result = model.evaluate(x_data, y_data)
print(np.argmax(y_predict))
# acc = accuracy_score(y_data, y_predict)

print(f'prediction: {y_predict}, acc: {result} and')

'''
For Single-layer Perceptron
    prediction: [[0.49975002] [0.726319  ] [0.69841415] [0.8601798 ]], acc: 0.8346884250640869 
  with batch_size=1, epochs=120
    prediction: [[0.5348199 ] [0.4400236 ] [0.2844896 ] [0.21368228]], acc: 0.7709275484085083
'''