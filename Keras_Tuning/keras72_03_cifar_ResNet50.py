from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import ResNet50
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

import tensorflow as tf
strategy = tf.distribute.MirroredStrategy(devices=['/gpu:1'])
with strategy.scope():
    rn50 = ResNet50(weights='imagenet', include_top=False,
                input_shape=(32, 32, 3))

    print('-----------------------------------------cifar10---------------------------------------------')
    print('Custom with ReduceLearningRate, entropy : 1.1582015752792358 , accuracy : 0.5856999754905701')

    model = Sequential()
    model.add(rn50)
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
    model.add(rn50)
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

    rn50.trainable = False

    model = Sequential()
    model.add(rn50)
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

    rn50.trainable = False

    model = Sequential()
    model.add(rn50)
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

import tensorflow as tf
strategy = tf.distribute.MirroredStrategy(devices=['/gpu:1'])
with strategy.scope():

    rn50 = ResNet50(weights='imagenet', include_top=False,
                input_shape=(32, 32, 3))

    print('-----------------------------------------cifar100---------------------------------------------')
    print('Custom-made vanilla CNN, entropy: 4.211299896240234 accuracy: 0.32280001044273376')

    model = Sequential()
    model.add(rn50)
    model.add(Flatten())
    # model.add(GlobalAveragePooling2D())
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(100, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    es = EarlyStopping(monitor='val_loss', patience=8, mode='min', verbose=2, restore_best_weights=True)
    hist = model.fit(x_train, y_train, epochs=300, batch_size=512, verbose=0, validation_split=0.25, callbacks=[es])
    loss = model.evaluate(x_test, y_test)

    print('Trainable=True & FC,', end='')
    print('entropy :', loss[0], ', accuracy :', loss[1])

    model = Sequential()
    model.add(rn50)
    # model.add(Flatten())
    model.add(GlobalAveragePooling2D())
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(100, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    es = EarlyStopping(monitor='val_loss', patience=8, mode='min', verbose=2, restore_best_weights=True)
    hist = model.fit(x_train, y_train, epochs=300, batch_size=512, verbose=0, validation_split=0.25, callbacks=[es])
    loss = model.evaluate(x_test, y_test)

    print('Trainable=True & GAP,', end='')
    print('entropy :', loss[0], ', accuracy :', loss[1])

    rn50.trainable = False

    model = Sequential()
    model.add(rn50)
    model.add(Flatten())
    # model.add(GlobalAveragePooling2D())
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(100, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    es = EarlyStopping(monitor='val_loss', patience=8, mode='min', verbose=2, restore_best_weights=True)
    hist = model.fit(x_train, y_train, epochs=300, batch_size=512, verbose=0, validation_split=0.25, callbacks=[es])
    loss = model.evaluate(x_test, y_test)

    print('Trainable=False & FC,', end='')
    print('entropy :', loss[0], ', accuracy :', loss[1])


    rn50.trainable = False

    model = Sequential()
    model.add(rn50)
    # model.add(Flatten())
    model.add(GlobalAveragePooling2D())
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(100, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    es = EarlyStopping(monitor='val_loss', patience=8, mode='min', verbose=2, restore_best_weights=True)
    hist = model.fit(x_train, y_train, epochs=300, batch_size=512, verbose=0, validation_split=0.25, callbacks=[es])
    loss = model.evaluate(x_test, y_test)

    print('Trainable=False & GAP,', end='')
    print('entropy :', loss[0], ', accuracy :', loss[1])

# -----------------------------------------cifar10---------------------------------------------
# Custom with ReduceLearningRate, entropy : 1.1582015752792358 , accuracy : 0.5856999754905701
# Trainable=True & FC,entropy : 1.038715124130249 , accuracy : 0.7860000133514404
# Trainable=True & GAP,entropy : 1.0740083456039429 , accuracy : 0.7757999897003174
# Trainable=False & FC,entropy : 1.0263761281967163 , accuracy : 0.7928000092506409
# Trainable=False & GAP,entropy : 1.0000519752502441 , accuracy : 0.7907000184059143
# -----------------------------------------cifar100---------------------------------------------
# Custom-made vanilla CNN, entropy: 4.211299896240234 accuracy: 0.32280001044273376
# Trainable=True & FC,entropy : 5.074321269989014 , accuracy : 0.014600000344216824
# Trainable=True & GAP,entropy : 3.2232000827789307 , accuracy : 0.46380001306533813
# Trainable=False & FC,entropy : 2.7026844024658203 , accuracy : 0.498199999332428
# Trainable=False & GAP,entropy : 2.658405303955078 , accuracy : 0.5037999749183655