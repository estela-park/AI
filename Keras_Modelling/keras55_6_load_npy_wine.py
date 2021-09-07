import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping


x_data = np.load('../_save/_npy/k55_x_data_wine.npy')
y_data = np.load('../_save/_npy/k55_y_data_wine.npy')

y_data = y_data.reshape(4898, 1)
enc = OneHotEncoder(handle_unknown='ignore')
y_data = enc.fit_transform(y_data).toarray()

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.85, random_state=72)

power_scaler = PowerTransformer()
x_train_pt = power_scaler.fit_transform(x_train)
x_test_pt = power_scaler.transform(x_test)

start = time.time()
model_pt = Sequential()
model_pt.add(Dense(64, input_shape=(11,), activation='relu'))
model_pt.add(Dense(128, activation='relu'))
model_pt.add(Dense(32, activation='relu'))
model_pt.add(Dense(7, activation='softmax'))

model_pt.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_accuracy', patience=24, mode='max', verbose=2, restore_best_weights=True)
model_pt.fit(x_train_pt, y_train, epochs=360, validation_split=0.15, verbose=2, batch_size=32, callbacks=[es])

loss_pt = model_pt.evaluate(x_test_pt, y_test)
predict_pt = model_pt.predict(x_test_pt)
end = time.time() - start

print('it took', end/60,'minutes and', end%60, 'seconds, accuracy:', loss_pt)

# it <Conv1D> took 30 seconds loss: 2.010981798171997, accuracy: 0.5877550840377808
# Conv1D                      loss: 1.590141177177429, accuracy: 0.5768707394599915
# DNN                         loss: 3.259600639343261, accuracy: 0.6503401398658752
# CNN                         loss: 1.162110567092895, accuracy: 0.5863945484161377
# LSTM                        loss: 1.095896959304809, accuracy: 0.5047619342803955