import time
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer

(x_train, y_train), (x_test, y_test) = cifar10.load_data() 
# (50000, 32, 32, 3) (50000, 1) (10000, 32, 32, 3) (10000, 1)

x_train = x_train.reshape(50000, 32*32*3)
x_test = x_test.reshape(10000, 32*32*3)

scaler_qt = QuantileTransformer()
x_train = scaler_qt.fit_transform(x_train)
x_test = scaler_qt.transform(x_test) 

x_train = x_train.reshape(50000, 32*32, 3)
x_test = x_test.reshape(10000, 32*32, 3)

enc = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

y_train = enc.fit_transform(y_train).toarray() # (50000, 10)
y_test = enc.transform(y_test).toarray()   # (10000, 10)


inputL = Input(shape=(32*32, 3))
hl = LSTM(units=16, activation='relu')(inputL)
hl = Dense(32, activation='selu')(hl)
hl = Dense(64, activation='relu')(hl)
hl = Dense(32, activation='selu')(hl)
hl = Dense(16, activation='selu')(hl)
outputL = Dense(10, activation='softmax')(hl)

model = Model(inputs=inputL, outputs=outputL)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
es = EarlyStopping(monitor='val_loss', mode='min', patience=4, verbose=2, restore_best_weights=True)

start = time.time()
model.fit(x_train, y_train, epochs=32, batch_size=128, verbose=2, validation_split=0.15, callbacks=[es])
end = time.time() - start

loss = model.evaluate(x_test, y_test)


'''
DNN w/o GAP
    entropy: 1.3354147672653198 accuracy: 0.5220000147819519
DNN w/h GAP
    entropy: 1.7635047435760498 accuracy: 0.36239999532699585
CNN  
    entropy: 1.5305525064468384 accuracy: 0.7447999715805054
LSTM: it took 5 hours 17 minutes and 2 seconds
    entropy: 2.0673 accuracy: 0.2489
'''