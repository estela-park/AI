from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, UpSampling2D, MaxPool2D
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
model.summary()
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
model.summary()
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
model.add(Dense(512, activation='selu'))
model.add(Dense(256, activation='selu'))
model.add(Dense(128, activation='selu'))
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
model.add(Dense(512, activation='selu'))
model.add(Dense(256, activation='selu'))
model.add(Dense(128, activation='selu'))
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
model.add(Dense(512, activation='selu'))
model.add(Dense(256, activation='selu'))
model.add(Dense(128, activation='selu'))
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
model.add(Dense(512, activation='selu'))
model.add(Dense(256, activation='selu'))
model.add(Dense(128, activation='selu'))
model.add(Dense(100, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=8, mode='min', verbose=2, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=300, batch_size=512, verbose=0, validation_split=0.25, callbacks=[es])
loss = model.evaluate(x_test, y_test)

print('Trainable=False & GAP,', end='')
print('entropy :', loss[0], ', accuracy :', loss[1])


'''
-----------------------------------------cifar10---------------------------------------------
Custom with ReduceLearningRate, entropy : 1.1582015752792358 , accuracy : 0.5856999754905701
'''