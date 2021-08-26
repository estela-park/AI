import time
from sklearn import datasets
from sklearn.preprocessing import RobustScaler
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Input

datasets = load_boston()

x = datasets.data   # (506, 13) 
y = datasets.target # (506,) 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, shuffle=True, random_state=78)
# (404, 13) (102, 13)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)


inputL = Input(shape=(13, 1))
hl = LSTM(units=24, activation='relu')(inputL)
hl = Dense(128, activation='relu')(hl)
hl = Dense(64, activation='relu')(hl)
hl = Dense(32, activation='relu')(hl)
hl = Dense(16, activation='relu')(hl)
outputL = Dense(1)(hl)

model = Model(inputs=inputL, outputs=outputL)

model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', mode='min', patience=12, verbose=2, restore_best_weights=True)

start = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=13                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc, verbose=2, validation_split=0.15, callbacks=[es])
end = time.time() - start

y_predict = model.predict([x_test])
r2 = r2_score(y_test, y_predict)

loss = model.evaluate(x_test, y_test)
print('training took',end,'seconds, loss : ', loss, 'R2 score : ', r2)

'''
Epoch 00215: early stopping
training took 79 seconds, loss :  16.927509307861328 R2 score :  0.7372268104863597
'''