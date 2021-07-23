from tensorflow.keras.datasets import cifar100
from sklearn.preprocessing import  RobustScaler, OneHotEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Input
from tensorflow.keras.callbacks import EarlyStopping
import time

(x_train, y_train), (x_test, y_test) = cifar100.load_data() 

x_train = x_train.reshape(50000, 32*32*3) # (50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32*32*3)   # (10000, 32, 32, 3)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 

x_train = x_train.reshape(50000, 32*32, 3)
x_test = x_test.reshape(10000, 32*32, 3)

enc = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
enc.fit(y_train)
y_train = enc.transform(y_train).toarray() # (50000, 100)
y_test = enc.transform(y_test).toarray()   # (10000, 100)

inputL = Input(shape=(32*32, 3))
hl = LSTM(units=20, activation='relu')(inputL)
hl = Dense(256, activation='relu')(hl)
hl = Dense(128, activation='relu')(hl)
hl = Dense(100, activation='relu')(hl)
outputL = Dense(100, activation='softmax')(hl)

model = Model(inputs=inputL, outputs=outputL)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')

es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)

start = time.time()
model.fit(x_train, y_train, epochs=30, batch_size=2000, verbose=2, validation_split=0.05)
end = time.time() - start

loss = model.evaluate(x_test, y_test)