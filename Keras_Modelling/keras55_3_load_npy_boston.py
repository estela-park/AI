import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import QuantileTransformer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input


x_data = np.load('../_save/_npy/k55_x_data_boston.npy')
y_data = np.load('../_save/_npy/k55_y_data_boston.npy')

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.85, random_state=82)

quantile_scaler = QuantileTransformer()
quantile_scaler.fit(x_train)
x_train_qt = quantile_scaler.transform(x_train)
x_test_qt = quantile_scaler.transform(x_test)

input1 = Input(shape=(13, ))
dense1 = Dense(104, activation='relu')(input1)
dense2 = Dense(52, activation='relu')(dense1)
dense3 = Dense(26, activation='relu')(dense2)
dense4 = Dense(13, activation='relu')(dense3)
output1 = Dense(1)(dense4)

model_qt = Model(inputs=input1, outputs=output1)

model_qt.compile(loss='mse', optimizer='adam')
model_qt.fit(x_train_qt, y_train, batch_size=13, epochs=300, verbose=0, validation_split=0.15)

loss_qt = model_qt.evaluate(x_test_qt, y_test)
predict_qt = model_qt.predict(x_test_qt)
r2_qt = r2_score(y_test, predict_qt)

print('loss:', loss_qt, 'R2 score:', r2_qt)

# this<simpleDNN>: loss: 11.98899555206298, R2 score: 0.8894193589970605
# LSTM             loss: 16.92750930786132, R2 score: 0.7372268104863597
# CNN              loss: 13.28035545349121, R2 score: 0.8775084879017759
# DNN              loss: 11.10914993286132, R2 score: 0.8975346251949831
# Conv1D           loss: 47.21376037597656, R2 score: 0.4754662342880125