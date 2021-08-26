import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, GlobalAvgPool1D, Dropout 
from tensorflow.keras.callbacks import EarlyStopping

dataset = pd.read_csv('../_data/winequality-white.csv', sep=';', header=0)

x = dataset.iloc[:, :11] # (4898, 11), DataFrame
x = np.array(x)
y = dataset['quality']   # (4898, )  , Series, [6 5 7 8 4 3 9]
y = np.array(y)          # [6 6 6 6 6 6 6 6 6 6 5 5 5 7 5 7 6 8 6 5], (4898, )


y = y.reshape(4898, 1)
enc = OneHotEncoder(handle_unknown='ignore')
y = enc.fit_transform(y).toarray() # <class 'scipy.sparse.csr.csr_matrix'>

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, random_state=72)

standard_scaler = StandardScaler()
standard_scaler.fit(x_train)
x_train_std = standard_scaler.transform(x_train)
x_train_std = x_train_std.reshape(x_train_std.shape[0], x_train_std.shape[1], 1)
x_test_std = standard_scaler.transform(x_test)
x_test_std = x_test_std.reshape(x_test_std.shape[0], x_test_std.shape[1], 1)

robust_scaler = RobustScaler()
robust_scaler.fit(x_train)
x_train_rb = robust_scaler.transform(x_train)
x_train_rb = x_train_rb.reshape(x_train_rb.shape[0], x_train_rb.shape[1], 1)
x_test_rb = robust_scaler.transform(x_test)
x_test_rb = x_test_rb.reshape(x_test_rb.shape[0], x_test_rb.shape[1], 1)

power_scaler = PowerTransformer()
power_scaler.fit(x_train)
x_train_pt = power_scaler.transform(x_train)
x_train_pt = x_train_pt.reshape(x_train_pt.shape[0], x_train_pt.shape[1], 1)
x_test_pt = power_scaler.transform(x_test)
x_test_pt = x_test_pt.reshape(x_test_pt.shape[0], x_test_pt.shape[1], 1)


start = time.time()
es = EarlyStopping(monitor='val_accuracy', patience=15, mode='max', verbose=2)

model_std = Sequential()
model_std.add(Conv1D(64, 2, input_shape=(11, 1), activation='relu', padding='same'))
model_std.add(Dropout(0.15))
model_std.add(Conv1D(128, 2, activation='relu'))
model_std.add(MaxPool1D())
model_std.add(Conv1D(32, 2, activation='relu'))
model_std.add(Dropout(0.15))
model_std.add(GlobalAvgPool1D())
model_std.add(Dense(7, activation='softmax'))

model_std.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_std.fit(x_train_std, y_train, epochs=360, validation_split=0.15, verbose=2, batch_size=32, callbacks=[es])

model_rb = Sequential()
model_rb.add(Conv1D(64, 2, input_shape=(11, 1), activation='relu', padding='same'))
model_rb.add(Dropout(0.15))
model_rb.add(Conv1D(128, 2, activation='relu'))
model_rb.add(MaxPool1D())
model_rb.add(Conv1D(32, 2, activation='relu'))
model_rb.add(Dropout(0.15))
model_rb.add(GlobalAvgPool1D())
model_rb.add(Dense(7, activation='softmax'))

model_rb.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_rb.fit(x_train_rb, y_train, epochs=360, validation_split=0.15, verbose=2, batch_size=32, callbacks=[es])

model_pt = Sequential()
model_pt.add(Conv1D(64, 2, input_shape=(11, 1), activation='relu', padding='same'))
model_pt.add(Dropout(0.15))
model_pt.add(Conv1D(128, 2, activation='relu'))
model_pt.add(MaxPool1D())
model_pt.add(Conv1D(32, 2, activation='relu'))
model_pt.add(Dropout(0.15))
model_pt.add(GlobalAvgPool1D())
model_pt.add(Dense(7, activation='softmax'))

model_pt.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_pt.fit(x_train_pt, y_train, epochs=360, validation_split=0.15, verbose=2, batch_size=32, callbacks=[es])

loss_std = model_std.evaluate(x_test_std, y_test)
predict_std = model_std.predict(x_test_std)

loss_rb = model_rb.evaluate(x_test_rb, y_test)
predict_rb = model_rb.predict(x_test_rb)

loss_pt = model_pt.evaluate(x_test_pt, y_test)
predict_pt = model_pt.predict(x_test_pt)

end = time.time() - start

print('it took', end/60,'minutes and', end%60, 'seconds')

print('for stdS, accuracy:', loss_std)
print('for robust, accuracy:', loss_rb)
print('for powerT, accuracy:', loss_pt)

'''
DNN- watching for train's accuracy
for stdS, accuracy: [3.30517840385437, 0.6204081773757935]
for robust, accuracy: [3.286365032196045, 0.6489796042442322]
for powerT, accuracy: [3.4653685092926025, 0.6693877577781677]
CNN- monitoring vtrain's accuracy
for stdS, accuracy: [1.1621105670928955, 0.5863945484161377]      
for robust, accuracy: [1.1812330484390259, 0.5836734771728516] *stopped at 303      
for powerT, accuracy: [1.0396184921264648, 0.5646258592605591] *stopped at 107
CNN- monitoring validation's accuracy
for stdS, accuracy: [1.0869648456573486, 0.5292516946792603]   *stopped at 63     
for robust, accuracy: [1.0778416395187378, 0.563265323638916]  *stopped at 122     
for powerT, accuracy: [1.0536588430404663, 0.5551020503044128] *stopped at 53

DNN performed much better.
'''