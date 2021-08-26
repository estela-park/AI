import time
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Dropout, GlobalAveragePooling1D, MaxPool1D
from tensorflow.keras.callbacks import EarlyStopping


datasets = load_iris()

x = datasets.data 
# (150, 4)
y = datasets.target 
# (150, )

y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=78)

scaler = MaxAbsScaler()
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test) 

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

model = Sequential()
model.add(Conv1D(filters=32, kernel_size=2, padding='same',    activation='relu', input_shape=(4, 1))) 
model.add(Dropout(0.2))
model.add(Conv1D(32, 2, padding='same', activation='relu'))
model.add(MaxPool1D())
model.add(Conv1D(64, 2, padding='same', activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dense(3, activation="softmax"))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
es = EarlyStopping(monitor='loss', patience=48, mode='min', verbose=2, restore_best_weights=True)
start = time.time()
model.fit(x_train, y_train, epochs=240, batch_size=32, verbose=2, validation_split=0.2, callbacks=[es])
end = time.time() - start

loss = model.evaluate(x_test, y_test)
print('it took',end//1,'seconds with loss:', loss[0], 'accuracy:', loss[1])

# it took 8.0 seconds with loss: 0.16817519068717957, accuracy: 0.9333333373069763