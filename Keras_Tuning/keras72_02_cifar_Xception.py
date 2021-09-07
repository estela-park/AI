# Really very heavy, V100 took 20minutes for one-time fitting

from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, UpSampling2D, MaxPool2D, Dropout
from tensorflow.keras.applications import Xception
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# (50000, 32, 32, 3) (50000, 1) (10000, 32, 32, 3) (10000, 1)

x_train = x_train.reshape(50000, 32 * 32 * 3)
x_test = x_test.reshape(10000, 32 * 32 * 3)
min_max_scaler = MinMaxScaler()
x_train = min_max_scaler.fit_transform(x_train)
x_test = min_max_scaler.transform(x_test)
x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)

enc = OneHotEncoder()
y_train = enc.fit_transform(y_train).toarray()
y_test = enc.transform(y_test).toarray()   

xcpt = Xception(weights='imagenet', include_top=False, input_shape=(96, 96, 3), classes=10)

print('-----------------------------------------cifar10---------------------------------------------')
print('Custom with ReduceLearningRate, entropy : 1.1582015752792358 , accuracy : 0.5856999754905701')

model = Sequential()
model.add(UpSampling2D((3, 3), input_shape=(32, 32, 3)))
model.add(xcpt)
model.add(MaxPool2D())
model.add(Flatten())
# model.add(GlobalAveragePooling2D())
model.add(Dense(256, activation='selu'))
model.add(Dense(128, activation='selu'))
model.add(Dense(32, activation='selu'))
model.add(Dense(10, activation='softmax'))
# Trainable params: 21,368,850
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=8, mode='min', verbose=2, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=300, batch_size=512, verbose=0, validation_split=0.25, callbacks=[es])
loss = model.evaluate(x_test, y_test)

print('Trainable=True & FC,', end='')
print('entropy :', loss[0], ', accuracy :', loss[1])

model = Sequential()
model.add(UpSampling2D((3, 3), input_shape=(32, 32, 3)))
model.add(xcpt)
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(256, activation='selu'))
model.add(Dense(128, activation='selu'))
model.add(Dense(32, activation='selu'))
model.add(Dense(10, activation='softmax'))
# Trainable params: 21,368,850
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=8, mode='min', verbose=2, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=300, batch_size=512, verbose=0, validation_split=0.25, callbacks=[es])
loss = model.evaluate(x_test, y_test)

print('Trainable=True & GAP,', end='')
print('entropy :', loss[0], ', accuracy :', loss[1])

xcpt.trainable = False

model = Sequential()
model.add(UpSampling2D((3, 3)))
model.add(xcpt)
model.add(Flatten())
# model.add(GlobalAveragePooling2D())
model.add(Dense(256, activation='selu'))
model.add(Dense(128, activation='selu'))
model.add(Dense(32, activation='selu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=8, mode='min', verbose=2, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=300, batch_size=512, verbose=0, validation_split=0.25, callbacks=[es])
loss = model.evaluate(x_test, y_test)

print('Trainable=False & FC,', end='')
print('entropy :', loss[0], ', accuracy :', loss[1])

xcpt.trainable = False

model = Sequential()
model.add(UpSampling2D((3, 3)))
model.add(xcpt)
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(256, activation='selu'))
model.add(Dense(128, activation='selu'))
model.add(Dense(32, activation='selu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=15, mode='min', verbose=2, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=300, batch_size=512, verbose=0, validation_split=0.25, callbacks=[es])
loss = model.evaluate(x_test, y_test)

print('Trainable=False & GAP,', end='')
print('entropy :', loss[0], ', accuracy :', loss[1])

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
# (50000, 32, 32, 3) (50000, 1) (10000, 32, 32, 3) (10000, 1)

x_train = x_train.reshape(50000, 32 * 32 * 3)
x_test = x_test.reshape(10000, 32 * 32 * 3)
min_max_scaler = MinMaxScaler()
x_train = min_max_scaler.fit_transform(x_train)
x_test = min_max_scaler.transform(x_test)
x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)

enc = OneHotEncoder()
y_train = enc.fit_transform(y_train).toarray()
y_test = enc.transform(y_test).toarray()   

xcpt = Xception(weights='imagenet', include_top=False,
              input_shape=(96, 96, 3))

print('-----------------------------------------cifar100---------------------------------------------')
print('Custom-made vanilla CNN, entropy: 4.211299896240234 accuracy: 0.32280001044273376')

model = Sequential()
model.add(UpSampling2D((3, 3)))
model.add(xcpt)
model.add(Flatten())
# model.add(GlobalAveragePooling2D())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='selu'))
model.add(Dense(100, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=8, mode='min', verbose=2, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=300, batch_size=512, verbose=0, validation_split=0.25, callbacks=[es])
loss = model.evaluate(x_test, y_test)

print('Trainable=True & FC,', end='')
print('entropy :', loss[0], ', accuracy :', loss[1])

model = Sequential()
model.add(UpSampling2D((3, 3)))
model.add(xcpt)
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='selu'))
model.add(Dense(100, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=8, mode='min', verbose=2, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=300, batch_size=512, verbose=0, validation_split=0.25, callbacks=[es])
loss = model.evaluate(x_test, y_test)

print('Trainable=True & GAP,', end='')
print('entropy :', loss[0], ', accuracy :', loss[1])

xcpt.trainable = False

model = Sequential()
model.add(UpSampling2D((3, 3)))
model.add(xcpt)
model.add(Flatten())
# model.add(GlobalAveragePooling2D())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='selu'))
model.add(Dense(100, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=8, mode='min', verbose=2, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=300, batch_size=512, verbose=0, validation_split=0.25, callbacks=[es])
loss = model.evaluate(x_test, y_test)

print('Trainable=False & FC,', end='')
print('entropy :', loss[0], ', accuracy :', loss[1])


xcpt.trainable = False

model = Sequential()
model.add(UpSampling2D((3, 3)))
model.add(xcpt)
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='selu'))
model.add(Dense(100, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=8, mode='min', verbose=2, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=300, batch_size=512, verbose=0, validation_split=0.25, callbacks=[es])
loss = model.evaluate(x_test, y_test)

print('Trainable=False & GAP,', end='')
print('entropy :', loss[0], ', accuracy :', loss[1])

# -----------------------------------------cifar10---------------------------------------------
# Custom with ReduceLearningRate, entropy : 1.1582015752792358 , accuracy : 0.5856999754905701
# Trainable=True & FC,entropy : 0.3734898567199707 , accuracy : 0.9187999963760376
# Trainable=True & GAP,entropy : 0.40902721881866455 , accuracy : 0.9143000245094299
# Trainable=False & FC,entropy : 0.5024968981742859 , accuracy : 0.9192000031471252
# Trainable=False & GAP,entropy : 0.3800846338272095 , accuracy : 0.9208999872207642
# -----------------------------------------cifar100---------------------------------------------
# Custom-made vanilla CNN, entropy: 4.211299896240234 accuracy: 0.32280001044273376
# Trainable=True & FC,entropy : 1.4891772270202637 , accuracy : 0.6514000296592712
# Trainable=True & GAP,entropy : 1.380493402481079 , accuracy : 0.6794000267982483
# Trainable=False & FC,entropy : 1.8204456567764282 , accuracy : 0.7116000056266785
# Trainable=False & GAP,entropy : 1.9364696741104126 , accuracy : 0.4966999888420105