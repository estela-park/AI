import time
import numpy as np
from sklearn.preprocessing import QuantileTransformer, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Dropout, MaxPool1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping


x_data = np.load('../_save/_npy/k55_x_data_mnist.npy')
y_data = np.load('../_save/_npy/k55_y_data_mnist.npy')

(x_train, y_train), (x_test, y_test) = (x_data[:60000], y_data[:60000]), (x_data[60000:], y_data[60000:])
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
x_train = x_train.reshape(60000, 28*28*1)
x_test = x_test.reshape(10000, 28*28*1)

scaler = QuantileTransformer()
x_train = scaler.fit_transform(x_train).reshape(60000, 28, 28)
x_test = scaler.transform(x_test).reshape(10000, 28, 28)

enc = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
y_train = enc.fit_transform(y_train).toarray() 
y_test = enc.transform(y_test).toarray()

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28, 28))) 
model.add(Dropout(0.25))
model.add(Conv1D(32, 2, padding='same', activation='relu'))                   
model.add(MaxPool1D())                                             
model.add(Flatten())                                              
model.add(Dense(124, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))

start = time.time()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
es = EarlyStopping(monitor='val_loss', patience=8, mode='min', verbose=2, restore_best_weights=True)
model.fit(x_train, y_train, epochs=240, batch_size=32, verbose=2, validation_split=0.2, callbacks=[es])
end = time.time() - start

loss = model.evaluate(x_test, y_test)
print('it took',end//1,'seconds with loss:', loss[0], 'accuracy:', loss[1])

# it took 2 mins and 10 secs entropy: 0.05366697907447815, accuracy: 0.985099971294403
# LSTM-Flatten               entropy: 0.08223655819892883, accuracy: 0.976800024509429
# LSTM-GAP                   entropy: 0.07211261242628098, accuracy: 0.978999972343444
# DNN4D w/o GAP              entropy: 0.19406241178512573, accuracy: 0.940599977970123
# DNN4D w/h GAP              entropy: 0.27129101753234863, accuracy: 0.914699971675872
# DNN3D w/o GAP              entropy: 0.06997454166412354, accuracy: 0.980700016021728
# CNN w/h GAP                entropy: 0.08116108924150467, accuracy: 0.984499990940094
# CNN                        entropy: 0.05236193165183067, accuracy: 0.991500020027160
# Conv1D-GAP                 entropy: 0.10824544727802277, accuracy: 0.968599975109100
# Conv1D-Flatten             entropy: 0.05419208481907844, accuracy: 0.985499978065490