from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
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
    ds121 = DenseNet121(weights='imagenet', include_top=False,
                input_shape=(32, 32, 3))

    print('-----------------------------------------cifar10---------------------------------------------')
    print('Custom with ReduceLearningRate, entropy : 1.1582015752792358 , accuracy : 0.5856999754905701')

    model = Sequential()
    model.add(ds121)
    model.add(Flatten())
    # model.add(GlobalAveragePooling2D())
    model.add(Dense(256, activation='selu'))
    model.add(Dense(128, activation='selu'))
    model.add(Dense(32, activation='selu'))
    model.add(Dense(10, activation='softmax'))

    opt = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])
    es = EarlyStopping(monitor='val_loss', patience=8, mode='min', verbose=2, restore_best_weights=True)
    hist = model.fit(x_train, y_train, epochs=300, batch_size=512, verbose=0, validation_split=0.25, callbacks=[es])
    loss = model.evaluate(x_test, y_test)

    print('Trainable=True & FC,', end='')
    print('entropy :', loss[0], ', accuracy :', loss[1])

    model = Sequential()
    model.add(ds121)
    # model.add(Flatten())
    model.add(GlobalAveragePooling2D())
    model.add(Dense(256, activation='selu'))
    model.add(Dense(128, activation='selu'))
    model.add(Dense(32, activation='selu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])
    es = EarlyStopping(monitor='val_loss', patience=8, mode='min', verbose=2, restore_best_weights=True)
    hist = model.fit(x_train, y_train, epochs=300, batch_size=512, verbose=0, validation_split=0.25, callbacks=[es])
    loss = model.evaluate(x_test, y_test)

    print('Trainable=True & GAP,', end='')
    print('entropy :', loss[0], ', accuracy :', loss[1])

    ds121.trainable = False

    model = Sequential()
    model.add(ds121)
    model.add(Flatten())
    # model.add(GlobalAveragePooling2D())
    model.add(Dense(256, activation='selu'))
    model.add(Dense(128, activation='selu'))
    model.add(Dense(32, activation='selu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])
    es = EarlyStopping(monitor='val_loss', patience=8, mode='min', verbose=2, restore_best_weights=True)
    hist = model.fit(x_train, y_train, epochs=300, batch_size=512, verbose=0, validation_split=0.25, callbacks=[es])
    loss = model.evaluate(x_test, y_test)

    print('Trainable=False & FC,', end='')
    print('entropy :', loss[0], ', accuracy :', loss[1])

    ds121.trainable = False

    model = Sequential()
    model.add(ds121)
    # model.add(Flatten())
    model.add(GlobalAveragePooling2D())
    model.add(Dense(256, activation='selu'))
    model.add(Dense(128, activation='selu'))
    model.add(Dense(32, activation='selu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])
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

    ds121 = DenseNet121(weights='imagenet', include_top=False,
                input_shape=(32, 32, 3))

    print('-----------------------------------------cifar100---------------------------------------------')
    print('Custom-made vanilla CNN, entropy: 4.211299896240234 accuracy: 0.32280001044273376')

    model = Sequential()
    model.add(ds121)
    model.add(Flatten())
    # model.add(GlobalAveragePooling2D())
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(100, activation='softmax'))

    opt = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])
    es = EarlyStopping(monitor='val_loss', patience=8, mode='min', verbose=2, restore_best_weights=True)
    hist = model.fit(x_train, y_train, epochs=300, batch_size=512, verbose=0, validation_split=0.25, callbacks=[es])
    loss = model.evaluate(x_test, y_test)

    print('Trainable=True & FC,', end='')
    print('entropy :', loss[0], ', accuracy :', loss[1])

    model = Sequential()
    model.add(ds121)
    # model.add(Flatten())
    model.add(GlobalAveragePooling2D())
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(100, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])
    es = EarlyStopping(monitor='val_loss', patience=8, mode='min', verbose=2, restore_best_weights=True)
    hist = model.fit(x_train, y_train, epochs=300, batch_size=512, verbose=0, validation_split=0.25, callbacks=[es])
    loss = model.evaluate(x_test, y_test)

    print('Trainable=True & GAP,', end='')
    print('entropy :', loss[0], ', accuracy :', loss[1])

    ds121.trainable = False

    model = Sequential()
    model.add(ds121)
    model.add(Flatten())
    # model.add(GlobalAveragePooling2D())
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(100, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])
    es = EarlyStopping(monitor='val_loss', patience=8, mode='min', verbose=2, restore_best_weights=True)
    hist = model.fit(x_train, y_train, epochs=300, batch_size=512, verbose=0, validation_split=0.25, callbacks=[es])
    loss = model.evaluate(x_test, y_test)

    print('Trainable=False & FC,', end='')
    print('entropy :', loss[0], ', accuracy :', loss[1])


    ds121.trainable = False

    model = Sequential()
    model.add(ds121)
    # model.add(Flatten())
    model.add(GlobalAveragePooling2D())
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(100, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])
    es = EarlyStopping(monitor='val_loss', patience=8, mode='min', verbose=2, restore_best_weights=True)
    hist = model.fit(x_train, y_train, epochs=300, batch_size=512, verbose=0, validation_split=0.25, callbacks=[es])
    loss = model.evaluate(x_test, y_test)

    print('Trainable=False & GAP,', end='')
    print('entropy :', loss[0], ', accuracy :', loss[1])

# -----------------------------------------cifar10---------------------------------------------
# Custom with ReduceLearningRate, entropy : 1.1582015752792358 , accuracy : 0.5856999754905701
# Trainable=True & FC,entropy : 0.7625563740730286 , accuracy : 0.8138999938964844
# Trainable=True & GAP,entropy : 0.8183514475822449 , accuracy : 0.8248000144958496
# Trainable=False & FC,entropy : 0.8735524415969849 , accuracy : 0.8363999724388123
# Trainable=False & GAP,entropy : 0.8895261883735657 , accuracy : 0.8399999737739563
# -----------------------------------------cifar100---------------------------------------------
# Custom-made vanilla CNN, entropy: 4.211299896240234 accuracy: 0.32280001044273376
# Trainable=True & FC,entropy : 2.238311529159546 , accuracy : 0.47269999980926514
# Trainable=True & GAP,entropy : 2.53848934173584 , accuracy : 0.5289000272750854
# Trainable=False & FC,entropy : 2.4027743339538574 , accuracy : 0.5580000281333923
# Trainable=False & GAP,entropy : 2.469766139984131 , accuracy : 0.5636000037193298