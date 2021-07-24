import time
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Dropout, MaxPool1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, OneHotEncoder

(x_train, y_train), (x_test, y_test) = mnist.load_data() 
# (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28*28*1)
x_test = x_test.reshape(10000, 28*28*1)

scaler = QuantileTransformer()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 

x_train = x_train.reshape(60000, 28, 28)
x_test = x_test.reshape(10000, 28, 28)

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

enc = OneHotEncoder()
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

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')

es = EarlyStopping(monitor='val_loss', patience=24, mode='min', verbose=1)

start = time.time()
model.fit(x_train, y_train, epochs=240, batch_size=256, verbose=2, validation_split=0.15, callbacks=[es])
end = time.time() - start

loss = model.evaluate(x_test, y_test)
print('it took', end, 'seconds with loss:', loss)