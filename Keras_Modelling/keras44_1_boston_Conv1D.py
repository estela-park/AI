import time
from sklearn.datasets import load_boston
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Dropout, GlobalAveragePooling1D, MaxPool1D
from tensorflow.keras.callbacks import EarlyStopping


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

es = EarlyStopping(monitor='val_loss', patience=48, mode='min', verbose=2, restore_best_weights=True)
start = time.time()
model.fit(x_train, y_train, epochs=300, batch_size=32, verbose=2, validation_split=0.15, callbacks=[es])
end = time.time() - start


loss = model.evaluate(x_test, y_test)
print('it took',end//1,'seconds with loss:', loss)
r2 = r2_score(y_test, model.predict(x_test))
print('R2 score:', r2)

# it<Conv1D> took 16 seconds with loss: 47.21376037597656, R2 score: 0.4754662342880125
# LSTM                            loss: 16.92750930786132, R2 score: 0.7372268104863597
# CNN                             loss: 13.28035545349121, R2 score: 0.8775084879017759
# DNN                             loss: 11.10914993286132, R2 score: 0.8975346251949831