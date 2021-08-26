import time
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Dropout, MaxPooling1D, GlobalAvgPool1D, LSTM
from tensorflow.keras.callbacks import EarlyStopping

dataset = load_breast_cancer()

# print(dataset.DESCR)
# print(dataset.feature_names)

x = dataset.data   
y = dataset.target 

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, random_state=78)

scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

model = Sequential()
model.add(LSTM(32, activation='relu', input_shape=(x_train.shape[1], 1), return_sequences=True)) 
model.add(Dropout(0.2))
model.add(Conv1D(32, 2, padding='same', activation='relu'))
model.add(MaxPooling1D())
model.add(GlobalAvgPool1D())
model.add(Dense(1, activation="sigmoid"))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics='acc')

es = EarlyStopping(monitor='val_loss', mode='min', patience=12, verbose=2, restore_best_weights=True)

start = time.time()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=240, validation_split=0.15, verbose=2, batch_size=32, callbacks=[es])
end = time.time() - start


loss = model.evaluate(x_test, y_test)
predict = model.predict(x_test)

print('it took', end/60, 'minutes and', end%60,'seconds')
print('for maxabs, accuracy:', loss)

'''
DNN
-for robust, accuracy: [0.41252321004867554, 0.9824561476707458]
CNN batch: 32, stopped early at 99, it took 18 seconds
-for maxabs, accuracy: [0.30689477920532227, 0.930232584476471]   
LSTM: stopped early at 74, it took 1 minute and 7 seconds
-for maxabs, accuracy: [0.9253537654876709, 0.8720930218696594]
'''