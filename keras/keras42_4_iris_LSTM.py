import time
from sklearn.datasets import load_iris
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, GlobalAvgPool1D, Dropout, LSTM
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

dataset = load_iris()

x = dataset.data   # (150, 4)
y = dataset.target # (150, )
y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85) # random_state=72

max_abs_scaler = MaxAbsScaler()
max_abs_scaler.fit(x_train)
x_train_ma = max_abs_scaler.transform(x_train)
x_test_ma = max_abs_scaler.transform(x_test)

x_train_ma = x_train_ma.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test_ma = x_test_ma.reshape(x_test.shape[0], x_test.shape[1], 1)

# if all the parameters are set to padding='valid', x_shape[1] becomes less than kernel size.
model_ma = Sequential()
model_ma.add(LSTM(32, input_shape=(4, 1), activation='relu', return_sequences=True))
model_ma.add(Dropout(0.15))
model_ma.add(Conv1D(128, kernel_size=2, activation='relu', padding='same'))
model_ma.add(MaxPool1D())
model_ma.add(GlobalAvgPool1D())
model_ma.add(Dense(3, activation='softmax'))

start = time.time()
es = EarlyStopping(monitor='val_loss', mode='min', patience=12, verbose=2, restore_best_weights=True)
model_ma.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_ma.fit(x_train_ma, y_train, epochs=240, validation_split=0.15, verbose=2, batch_size=32, callbacks=[es])
end = time.time() - start

loss_ma = model_ma.evaluate(x_test_ma, y_test)
predict_ma = model_ma.predict(x_test_ma)

print('it took', end/60, 'minutes and', end%60,'seconds')
print('for maxabs, accuracy:', loss_ma)

'''
DNN
-for maxabs, accuracy: [0.037038467824459076, 1.0]
CNN stopped early at 54, any other hyper-parameters are set to match
-for maxabs, accuracy: [0.1542966216802597, 0.9130434989929199]   
LSTM: stopped early at 104 took 10 seconds
-for maxabs, accuracy: [0.09124008566141129, 0.95652174949646]
'''