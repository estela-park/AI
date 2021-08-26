import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, GlobalAvgPool1D, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping

dataset = pd.read_csv('../_data/white_wine.csv', sep=';', header=0)

x = dataset.iloc[:, :11] # (4898, 11), DataFrame
x = np.array(x)
y = dataset['quality']   # (4898, )  , Series, [6 5 7 8 4 3 9]
y = np.array(y)          # [6 6 6 6 6 6 6 6 6 6 5 5 5 7 5 7 6 8 6 5], (4898, )


y = y.reshape(4898, 1)
enc = OneHotEncoder(handle_unknown='ignore')
y = enc.fit_transform(y).toarray() # <class 'scipy.sparse.csr.csr_matrix'>

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, random_state=72)

robust_scaler = RobustScaler()
robust_scaler.fit(x_train)
x_train_rb = robust_scaler.transform(x_train)
x_train_rb = x_train_rb.reshape(x_train_rb.shape[0], x_train_rb.shape[1], 1)
x_test_rb = robust_scaler.transform(x_test)
x_test_rb = x_test_rb.reshape(x_test_rb.shape[0], x_test_rb.shape[1], 1)

start = time.time()
es = EarlyStopping(monitor='val_loss', mode='min', patience=12, verbose=2, restore_best_weights=True)

model_rb = Sequential()
model_rb.add(LSTM(32, input_shape=(11, 1), activation='relu', return_sequences=True))
model_rb.add(Dropout(0.15))
model_rb.add(Conv1D(128, 2, activation='relu'))
model_rb.add(MaxPool1D())
model_rb.add(GlobalAvgPool1D())
model_rb.add(Dense(7, activation='softmax'))

model_rb.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_rb.fit(x_train_rb, y_train, epochs=360, validation_split=0.15, verbose=2, batch_size=32, callbacks=[es])

loss_rb = model_rb.evaluate(x_test_rb, y_test)
predict_rb = model_rb.predict(x_test_rb)

end = time.time() - start

print('it took', end/60,'minutes and', end%60, 'seconds')
print('for robust, accuracy:', loss_rb)


'''
DNN
-for powerT, accuracy: [3.2596006393432617, 0.6503401398658752]
CNN
-for stdS, accuracy: [1.1621105670928955, 0.5863945484161377] 
LSTM: ES @72 , 3 mins & 40 secs
-for robust, accuracy: [1.0958969593048096, 0.5047619342803955]
'''