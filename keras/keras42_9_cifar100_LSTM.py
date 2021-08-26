import time
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import  RobustScaler, OneHotEncoder


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
hl = LSTM(units=16, activation='selu')(inputL)
hl = Dense(32, activation='selu')(hl)
hl = Dense(64, activation='selu')(hl)
hl = Dense(128, activation='relu')(hl)
hl = Dense(256, activation='relu')(hl)
hl = Dense(128, activation='selu')(hl)
outputL = Dense(100, activation='softmax')(hl)

model = Model(inputs=inputL, outputs=outputL)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')
es = EarlyStopping(monitor='val_loss', mode='min', patience=4, verbose=2, restore_best_weights=True)

start = time.time()
model.fit(x_train, y_train, epochs=32, batch_size=128, verbose=2, validation_split=0.15, callbacks=[es])
end = time.time() - start

loss = model.evaluate(x_test, y_test)
print('time spent:',end//60,'minutes and', end%60, 'seconds')
print('entropy:', loss, 'accuracy:', loss)
'''
DNN w/o GAP
    entropy: 3.238474130630493 accuracy: 0.22609999775886536
DNN w/h GAP
    entropy: 3.914315938949585 accuracy: 0.1014999970793724
CNN
    entropy: 4.211299896240234 accuracy: 0.32280001044273376
LSTM it took 2 hours 16 minutes
    entropy: 4.2478 accuracy: 0.0528
'''
