import time
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, MaxPooling1D, Reshape, Conv2D, MaxPool2D
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers.convolutional import Conv2D

(x_train, y_train), (x_test, y_test) = mnist.load_data() # (60000, 28, 28), (60000,), (10000, 28, 28), (10000,): [0 1 2 3 4 5 6 7 8 9]

enc = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
y_train = enc.fit_transform(y_train).toarray() 
y_test = enc.transform(y_test).toarray()

model = Sequential()
model.add(Dense(units=32, input_shape=(28, 28), activation='relu'))  
model.add(Dropout(0.15))             
model.add(Flatten())    
model.add(Dense(784, activation='relu'))                                                                                 
model.add(Reshape((28, 28, 1)))
model.add(Conv2D(32, (2, 2)))   
model.add(MaxPool2D())                                                       
model.add(Flatten())                                                               
model.add(Dense(64, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 28, 32)            928
_________________________________________________________________
dropout (Dropout)            (None, 28, 32)            0
_________________________________________________________________
max_pooling1d (MaxPooling1D) (None, 14, 32)            0
_________________________________________________________________
flatten (Flatten)            (None, 448)               0
_________________________________________________________________
dense_1 (Dense)              (None, 784)               352016
_________________________________________________________________
reshape (Reshape)            (None, 28, 28, 1)         0
_________________________________________________________________
conv2d (Conv2D)              (None, 27, 27, 32)        160
_________________________________________________________________
dropout_1 (Dropout)          (None, 27, 27, 32)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 26, 26, 64)        8256
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 13, 13, 64)        0
_________________________________________________________________
flatten_1 (Flatten)          (None, 10816)             0
_________________________________________________________________
dense_2 (Dense)              (None, 64)                692288
_________________________________________________________________
dense_3 (Dense)              (None, 16)                1040
_________________________________________________________________
dense_4 (Dense)              (None, 10)                170
=================================================================
Total params: 1,054,858
Trainable params: 1,054,858
Non-trainable params: 0
_________________________________________________________________
'''

es = EarlyStopping(monitor='val_loss', patience=18, mode='min', verbose=2)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

start = time.time()
model.fit(x_train, y_train, epochs=120, verbose=2, validation_split=0.15, batch_size=256, callbacks=[es])
end = time.time() - start

loss = model.evaluate(x_test, y_test)

print('it took', end/60, 'minutes and', end%60,'seconds')
print('entropy:', loss[0],'accuracy:', loss[1])

'''
it took 1 minute and 54 seconds
entropy: 2.3009989261627197 accuracy: 0.11349999904632568
-performance too bad, deleted one Conv2D layer
it took 33 seconds
entropy: 0.1320798248052597 accuracy: 0.9751999974250793
'''