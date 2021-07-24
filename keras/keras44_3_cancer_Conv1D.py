import time
from sklearn.preprocessing import MaxAbsScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Dropout, GlobalAveragePooling1D, MaxPool1D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

datasets = load_breast_cancer()

x = datasets.data 
# (569, 30) 
y = datasets.target 
# (569,)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=76)

scaler = MaxAbsScaler()

x_train = scaler.fit_transform(x_train) 
x_test = scaler.transform(x_test) 

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

model = Sequential()
model.add(Conv1D(filters=128, kernel_size=2, padding='same', activation='relu', input_shape=(30, 1))) 
model.add(Dropout(0.2))
model.add(Conv1D(32, 2, padding='same', activation='relu'))
model.add(MaxPool1D())
model.add(Conv1D(8, 2, padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(MaxPool1D())
model.add(GlobalAveragePooling1D())
model.add(Dense(1, activation="sigmoid"))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics='acc')

es = EarlyStopping(monitor='loss', patience=20, mode='min', verbose=1)

start = time.time()
model.fit(x_train, y_train, epochs=240, batch_size=32, verbose=2, validation_split=0.2, callbacks=[es])
end = time.time() - start

loss = model.evaluate(x_test, y_test)
print('it took',end,'seconds with loss:',loss)