import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import QuantileTransformer


x_data = np.load('../_save/_npy/k55_x_data_diabetes.npy')
y_data = np.load('../_save/_npy/k55_y_data_diabetes.npy')

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.8, random_state=72)

quantile_scaler = QuantileTransformer()
quantile_scaler.fit(x_train)
x_train_qt = quantile_scaler.transform(x_train)
x_test_qt = quantile_scaler.transform(x_test)

start = time.time()
model_qt = Sequential()
model_qt.add(Dense(64, input_shape=(10,), activation='relu'))
model_qt.add(Dense(128, activation='relu'))
model_qt.add(Dense(32, activation='relu'))
model_qt.add(Dense(1))

model_qt.compile(loss='mse', optimizer='adam', metrics=['mae'])
model_qt.fit(x_train_qt, y_train, epochs=240, validation_split=0.15, verbose=0, batch_size=32)

loss_qt = model_qt.evaluate(x_test_qt, y_test)
predict_qt = model_qt.predict(x_test_qt)
end = time.time() - start

r2_qt = r2_score(y_test, predict_qt)

print('it took',end, 'loss:', loss_qt, 'accuracy:', r2_qt)

# this(DNN) took 10 secs loss: 2319.841552734, R2 score: 0.648680204176222
# Conv1D                 loss: 3652.457275390, R2 score: 0.385958354247822
# LSTM                   loss: 5206.095703125, R2 score: 0.211582105080756
# CNN                    loss: 3084.109130859, R2 score: 0.532938514883232
# DNN                    loss: 2109.333251953, R2 score: 0.680559819627763