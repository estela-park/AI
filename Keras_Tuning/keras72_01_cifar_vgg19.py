# VGG16 worked without activation fn applied to FC,
# but VGG19 frozed without activation, so 
#  > selu-selu-None-softmax: worked
#  > selu-selu-relu-softmax: frozed
#  > selu-selu-selu-softmax: worked
#
# even with selu*3, model frozed with 100 classification: when trainable=False, it worked.
# FC1: 512 -> 256           X
# FC: one layer deeper      X
# FC: 1028, 256, 128, 100   ^
# FC: 2046, 512, 128, 100   O


from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG19
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

vgg19 = VGG19(weights='imagenet', include_top=False,
              input_shape=(32, 32, 3))

print('-----------------------------------------cifar10---------------------------------------------')
print('Custom with ReduceLearningRate, entropy : 1.1582015752792358 , accuracy : 0.5856999754905701')

model = Sequential()
model.add(vgg19)
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

print('Trainable=True & FC,', end='')
print('entropy :', loss[0], ', accuracy :', loss[1])

model = Sequential()
model.add(vgg19)
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(256, activation='selu'))
model.add(Dense(128, activation='selu'))
model.add(Dense(32, activation='selu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=8, mode='min', verbose=2, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=300, batch_size=512, verbose=0, validation_split=0.25, callbacks=[es])
loss = model.evaluate(x_test, y_test)

print('Trainable=True & GAP,', end='')
print('entropy :', loss[0], ', accuracy :', loss[1])

vgg19.trainable = False

model = Sequential()
model.add(vgg19)
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

vgg19.trainable = False

model = Sequential()
model.add(vgg19)
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

vgg19 = VGG19(weights='imagenet', include_top=False,
              input_shape=(32, 32, 3))

print('-----------------------------------------cifar100---------------------------------------------')
print('Custom-made vanilla CNN, entropy: 4.211299896240234 accuracy: 0.32280001044273376')

model = Sequential()
model.add(vgg19)
model.add(Flatten())
# model.add(GlobalAveragePooling2D())
model.add(Dense(1028, activation='selu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='selu'))
model.add(Dense(100, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=8, mode='min', verbose=2, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=300, batch_size=512, verbose=0, validation_split=0.25, callbacks=[es])
loss = model.evaluate(x_test, y_test)

print('Trainable=True & FC,', end='')
print('entropy :', loss[0], ', accuracy :', loss[1])

model = Sequential()
model.add(vgg19)
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(2046, activation='selu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(128, activation='selu'))
model.add(Dense(100, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=8, mode='min', verbose=2, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=300, batch_size=512, verbose=0, validation_split=0.25, callbacks=[es])
loss = model.evaluate(x_test, y_test)

print('Trainable=True & GAP,', end='')
print('entropy :', loss[0], ', accuracy :', loss[1])

vgg19.trainable = False

model = Sequential()
model.add(vgg19)
model.add(Flatten())
# model.add(GlobalAveragePooling2D())
model.add(Dense(2046, activation='selu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(128, activation='selu'))
model.add(Dense(100, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=8, mode='min', verbose=2, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=300, batch_size=512, verbose=0, validation_split=0.25, callbacks=[es])
loss = model.evaluate(x_test, y_test)

print('Trainable=False & FC,', end='')
print('entropy :', loss[0], ', accuracy :', loss[1])


vgg19.trainable = False

model = Sequential()
model.add(vgg19)
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(2046, activation='selu'))
model.add(Dense(512, activation='selu'))
model.add(Dense(128, activation='relu'))
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
Trainable=True & FC,entropy : 0.7751458287239075 , accuracy : 0.7774999737739563
Trainable=True & GAP,entropy : 0.7542965412139893 , accuracy : 0.7771000266075134 
Trainable=False & FC,entropy : 0.7781859040260315 , accuracy : 0.7846999764442444
Trainable=False & GAP,entropy : 0.7760912179946899 , accuracy : 0.7843000292778015


-----------------------------------------cifar100---------------------------------------------
Custom-made vanilla CNN, entropy: 4.211299896240234 accuracy: 0.32280001044273376
Trainable=True & FC,entropy : 2.9209189414978027 , accuracy : 0.2777999937534332
Trainable=True & GAP,entropy : 2.838899612426758 , accuracy : 0.28600001335144043
slim, Trainable=False & FC,entropy : 3.1639835834503174 , accuracy : 0.24809999763965607
slim, Trainable=False & GAP,entropy : 3.1672420501708984 , accuracy : 0.24400000274181366
expanded, Trainable=False & FC,entropy : 2.720061779022217 , accuracy : 0.3255999982357025
expanded, Trainable=False & GAP,entropy : 2.76810622215271 , accuracy : 0.320499986410141
'''