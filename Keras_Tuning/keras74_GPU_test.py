# making sure the every GPU work
#
# Distributed operation should be implemented while building model

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus :
    try :
        tf.config.experimental.list_physical_devices(gpus[0], 'GPU')
        print('gpu is set')

    except RuntimeError as e:
        print(e)

exit()

import time
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, MaxPool2D, Dropout, GlobalAveragePooling2D, Layer, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


strategy = tf.distribute.MirroredStrategy()

# 1. data-set
(x_train, y_train), (x_test, y_test) = cifar10.load_data() 
# x: (50000, 32, 32, 3) (10000, 32, 32, 3)
# y:  (50000, 1) (10000, 1)

x_train = x_train.reshape(-1, 28, 28, 1).astype('float32')/255
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32')/255


enc = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
y_train = enc.fit_transform(y_train).toarray()
y_test = enc.transform(y_test).toarray()

# 2. modeling
#    compile together
with strategy.scope():
    input_l = Input(shape=(32, 32, 3))
    hl = Dense(32, activation='relu')(input_l)
    hl = Dropout(0.2)(hl)
    hl = Dense(64, activation='relu')(hl)
    hl = MaxPool2D()(hl)
    hl = Dense(32, activation='relu')(hl)
    hl = Dropout(0.2)(hl)
    hl = Dense(64, activation='relu')(hl)
    hl = MaxPool2D()(hl)                         
    hl = Dense(32, activation='relu')(hl)
    hl = Dropout(0.2)(hl)
    hl = Dense(64, activation='relu')(hl)
    hl = MaxPool2D()(hl)
    hl1 = Flatten()(hl)
    output_1 = Dense(10, activation='softmax')(hl1) # KerasTensor (None, 10)

    hl2 = GlobalAveragePooling2D()(hl)
    hl2 = Dense(10, activation='softmax')(hl2)
    output_2 = Layer()(hl2)                         # KerasTensor (None, 10)

    # 3. compilation & training
    model = Model(inputs=[input_l], outputs=[output_1, output_2])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    model.fit(x_train, y_train, epochs=100, batch_size=64, verbose=2, validation_split=0.15, callbacks=[es])

    # 4. evaluation
    loss = model.evaluate(x_test, y_test)
    print('global_ entropy:', loss[0])
    print('entropy:', loss[1],'accuracy:', loss[2])
    print('entropy:', loss[3],'accuracy:', loss[4])