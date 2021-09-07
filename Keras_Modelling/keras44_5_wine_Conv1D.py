import time
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Dropout, GlobalAveragePooling1D, MaxPool1D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

datasets = pd.read_csv('../_data/white_wine.csv', sep=';', index_col=None, header=0 ) 
# (4898, 12)

x = datasets.iloc[:, 0:11] 
# (4898, 11)
y = datasets.iloc[:, [11]] 
# (4898, 10)

enc = OneHotEncoder()
y = enc.fit_transform(y).toarray()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=24)

scaler = StandardScaler(with_std=False)
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test) 

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

model = Sequential()
model.add(Conv1D(filters=32, kernel_size=2, padding='same', activation='relu', input_shape=(11, 1))) 
model.add(Dropout(0.1))
model.add(Conv1D(32, 2, padding='same', activation='relu'))
model.add(MaxPool1D())
model.add(Conv1D(64, 2, padding='same', activation='relu'))
model.add(Dropout(0.1))
model.add(Conv1D(64, 2, padding='same', activation='relu'))
model.add(MaxPool1D())
model.add(Conv1D(128, 2, padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv1D(128, 2, padding='same', activation='relu'))
model.add(MaxPool1D())
model.add(Conv1D(256, 2, padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv1D(256, 2, padding='same', activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dense(7, activation="softmax"))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
es = EarlyStopping(monitor='loss', patience=48, mode='min', verbose=2, restore_best_weights=True)
start = time.time()
model.fit(x_train, y_train, epochs=240, batch_size=32, verbose=2, validation_split=0.2, callbacks=[es])
end = time.time() - start

loss = model.evaluate(x_test, y_test)
print('it took',end//1,'seconds with loss:', loss[0], 'accuracy:', loss[1])

# Conv1D this took 3 minutes with loss: 1.5901411771774292, accuracy: 0.5768707394599915
# DNN                             loss: 3.2596006393432617, accuracy: 0.6503401398658752
# CNN                             loss: 1.1621105670928955, accuracy: 0.5863945484161377
# LSTM                            loss: 1.0958969593048096, accuracy: 0.5047619342803955