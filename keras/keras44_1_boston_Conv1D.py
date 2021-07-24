import time
from sklearn.preprocessing import MaxAbsScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Dropout, GlobalAveragePooling1D, MaxPool1D
from sklearn.datasets import load_boston
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split


datasets = load_boston()

x = datasets.data 
# (506, 13)
y = datasets.target 
# (506,)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, shuffle=True, random_state=78)
# (404, 13) (102, 13)

scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

model = Sequential()
model.add(Conv1D(filters=32, kernel_size=2, padding='same', activation='relu', input_shape=(13, 1))) 
model.add(Dropout(0.25))
model.add(Conv1D(32, 2, padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(MaxPool1D())
model.add(Conv1D(128, 2, padding='same', activation='relu'))
model.add(MaxPool1D())
model.add(GlobalAveragePooling1D())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1)
start = time.time()
model.fit(x_train, y_train, epochs=300, batch_size=32, verbose=2, validation_split=0.15, callbacks=[es])
end = time.time() - start

y_predict = model.predict([x_test])

loss = model.evaluate(x_test, y_test)
print('it took',end,'seconds with loss:',loss)