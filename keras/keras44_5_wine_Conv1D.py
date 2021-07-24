import time
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Dropout, GlobalAveragePooling1D, MaxPool1D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

datasets = pd.read_csv('../_data/winequality-white.csv', sep=';', index_col=None, header=0 ) 
# (4898, 12)

x = datasets.iloc[:, 0:11] 
# (4898, 11)
y = datasets.iloc[:, [11]] 
# (4898, 10)

enc = OneHotEncoder()
y = enc.fit_transform(y).toarray()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=24)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
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

es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)

start = time.time()
hist = model.fit(x_train, y_train, epochs=240, batch_size=128, verbose=2,  validation_split=0.15, callbacks=[es])
end = time.time() - start

loss = model.evaluate(x_test, y_test)
print('it took',end,'seconds with loss:',loss)