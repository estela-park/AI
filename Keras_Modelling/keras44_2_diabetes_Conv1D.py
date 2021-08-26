import time
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Dropout, GlobalAveragePooling1D, MaxPool1D
from tensorflow.keras.callbacks import EarlyStopping



datasets = load_diabetes()

x = datasets.data 
# (442, 10)
y = datasets.target 
# (442,)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, shuffle=True, random_state=72)

scaler = MaxAbsScaler()
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test) 

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(10, 1))) 
model.add(Dropout(0.2))
model.add(Conv1D(128, 2, padding='valid', activation='relu'))
model.add(MaxPool1D())
model.add(Conv1D(32, 2, padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(MaxPool1D())
model.add(GlobalAveragePooling1D())
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='val_loss', patience=48, mode='min', verbose=2, restore_best_weights=True)

start = time.time()
model.fit(x_train, y_train, epochs=240, batch_size=32, verbose=2, validation_split=0.15, callbacks=[es])
end = time.time() - start

loss = model.evaluate(x_test, y_test)
print('it took',end//1,'seconds with loss:', loss)
r2 = r2_score(y_test, model.predict(x_test))
print('R2 score:', r2)

# it took 12 seconds with loss: 3652.457275390625
# R2 score: 0.385958354247822