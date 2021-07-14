from sklearn.datasets import load_boston
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np

dataset = load_boston()
scaler = MinMaxScaler()
x = dataset.data   # min: 0.0, max: 711.0
# x = (x - np.min(x)) / (np.max(x) - np.min(x))
scaler.fit(x) # x is not altered, the model is fitted to x.
x_scaled = scaler.transform(x)
# print(np.min(x_scaled), np.max(x_scaled))
y = dataset.target # min: 5.0, max: 50.0
# y = (y - np.min(y)) / (np.max(y) - np.min(y))

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, train_size=0.85, random_state=82)

input1 = Input(shape=(13, ))
hl = Dense(128, activation='relu')(input1)
hl = Dense(64, activation='relu')(hl)
hl = Dense(64, activation='relu')(hl)
hl = Dense(64, activation='relu')(hl)
hl = Dense(32, activation='relu')(hl)
output1 = Dense(1)(hl)

model = Model(inputs=input1, outputs=output1)

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, batch_size=13, epochs=320, verbose=0, validation_split=0.15)

loss = model.evaluate(x_test, y_test)
predict = model.predict(x_test)
r2 = r2_score(y_test, predict)

print('loss:', loss, 'r2:', r2)

'''
with minmax_as a whole x
loss: 0.008752845227718353 r2: 0.8365177565943386
with minmax scaler fn
loss: 10.06490707397461 r2: 0.90716621203317
'''