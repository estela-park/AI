import time
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


x_data = np.load('../_save/_npy/k55_x_data_cifar10.npy')
y_data = np.load('../_save/_npy/k55_y_data_cifar10.npy')

(x_train, y_train), (x_test, y_test) = (x_data[:50000], y_data[:50000]), (x_data[50000:], y_data[50000:])

x_train = x_train.reshape(50000, 32 * 32 * 3)
x_test = x_test.reshape(10000, 32 * 32 * 3)
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)

enc = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
y_train = enc.fit_transform(y_train).toarray()
y_test = enc.transform(y_test).toarray()

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(32, 32, 3))) 
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))                   
model.add(MaxPool2D())                                         
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))                   
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))    
model.add(MaxPool2D())                                         
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))                   
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPool2D())     
model.add(Flatten())                                              
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

start = time.time()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_acc', patience=20, mode='max', verbose=2)
model.fit(x_train, y_train, epochs=100, batch_size=64, verbose=2, validation_split=0.15, callbacks=[es])
end = time.time() - start

loss = model.evaluate(x_test, y_test)
print('it took', end/60, 'minutes and', end%60,'seconds, entropy:', loss[0],'accuracy:', loss[1])

# it took 2 mins 58 secs, entropy: 1.7034, accuracy: 0.744
# DNN w/o GAP             entropy: 1.3354, accuracy: 0.522
# DNN w/h GAP             entropy: 1.7635, accuracy: 0.362
# CNN                     entropy: 1.5305, accuracy: 0.744
# Conv1D                  entropy: 1.2629, accuracy: 0.547
# LSTM                    entropy: 2.0673, accuracy: 0.248