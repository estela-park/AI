from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import ResNet101
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

    rn101 = ResNet101(weights='imagenet', include_top=False,
                input_shape=(32, 32, 3))

    print('-----------------------------------------cifar10---------------------------------------------')
    print('Custom with ReduceLearningRate, entropy : 1.1582015752792358 , accuracy : 0.5856999754905701')

    model = Sequential()
    model.add(rn101)
    model.add(Flatten())
    # model.add(GlobalAveragePooling2D())
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    es = EarlyStopping(monitor='val_loss', patience=8, mode='min', verbose=2, restore_best_weights=True)
    hist = model.fit(x_train, y_train, epochs=300, batch_size=512, verbose=0, validation_split=0.25, callbacks=[es])
    loss = model.evaluate(x_test, y_test)

    print('Trainable=True & FC,', end='')
    print('entropy :', loss[0], ', accuracy :', loss[1])

    model = Sequential()
    model.add(rn101)
    # model.add(Flatten())
    model.add(GlobalAveragePooling2D())
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    es = EarlyStopping(monitor='val_loss', patience=8, mode='min', verbose=2, restore_best_weights=True)
    hist = model.fit(x_train, y_train, epochs=300, batch_size=512, verbose=0, validation_split=0.25, callbacks=[es])
    loss = model.evaluate(x_test, y_test)

    print('Trainable=True & GAP,', end='')
    print('entropy :', loss[0], ', accuracy :', loss[1])

    rn101.trainable = False

    model = Sequential()
    model.add(rn101)
    model.add(Flatten())
    # model.add(GlobalAveragePooling2D())
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    es = EarlyStopping(monitor='val_loss', patience=8, mode='min', verbose=2, restore_best_weights=True)
    hist = model.fit(x_train, y_train, epochs=300, batch_size=512, verbose=0, validation_split=0.25, callbacks=[es])
    loss = model.evaluate(x_test, y_test)

    print('Trainable=False & FC,', end='')
    print('entropy :', loss[0], ', accuracy :', loss[1])

    rn101.trainable = False

    model = Sequential()
    model.add(rn101)
    # model.add(Flatten())
    model.add(GlobalAveragePooling2D())
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(256, activation='relu'))
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

    rn101 = ResNet101(weights='imagenet', include_top=False,
                input_shape=(32, 32, 3))

    print('-----------------------------------------cifar100---------------------------------------------')
    print('Custom-made vanilla CNN, entropy: 4.211299896240234 accuracy: 0.32280001044273376')

    model = Sequential()
    model.add(rn101)
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
    model.add(rn101)
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

    rn101.trainable = False

    model = Sequential()
    model.add(rn101)
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


    rn101.trainable = False

    model = Sequential()
    model.add(rn101)
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
# Trainable=True & FC,entropy : 1.0767325162887573 , accuracy : 0.7803000211715698
# Trainable=True & GAP,entropy : 1.113024115562439 , accuracy : 0.741599977016449
# Trainable=False & FC,entropy : 1.1748497486114502 , accuracy : 0.7786999940872192
# Trainable=False & GAP,entropy : 1.2003976106643677 , accuracy : 0.7803000211715698
# -----------------------------------------cifar100---------------------------------------------
# Custom-made vanilla CNN, entropy: 4.211299896240234 accuracy: 0.32280001044273376
# Trainable=True & FC,entropy : 3.3476784229278564 , accuracy : 0.46799999475479126
# Trainable=True & GAP,entropy : 3.3160200119018555 , accuracy : 0.4413999915122986
# Trainable=False & FC,entropy : 3.118589401245117 , accuracy : 0.47360000014305115
# Trainable=False & GAP,entropy : 3.1022121906280518 , accuracy : 0.4778999984264374