from tensorflow.keras.datasets import cifar10
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, 
from tensorflow.keras.models import  Model
from tensorflow.keras.layers import Dense, LSTM, Input
from tensorflow.keras.callbacks import EarlyStopping

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

y_train = enc.transform(y_train).toarray() # (50000, 10)
y_test = enc.transform(y_test).toarray()   # (10000, 10)


inputL = Input(shape=(32*32, 3))
hl = LSTM(units=10, activation='relu')(inputL)
hl = Dense(128, activation='relu')(hl)
hl = Dense(64, activation='relu')(hl)
hl = Dense(32, activation='relu')(hl)
hl = Dense(16, activation='relu')(hl)
outputL = Dense(10, activation='softmax')(hl)

model = Model(inputs=inputL, outputs=outputL)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)

start = time.time()
model.fit(x_train, y_train, epochs=10, batch_size=2000, verbose=2, validation_split=0.05)
end = time.time() - start

loss = model.evaluate(x_test, y_test)