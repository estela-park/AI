# 3D DNN (keras32: 4D DNN)

import time
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, MaxPooling1D
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping

(x_train, y_train), (x_test, y_test) = mnist.load_data() # (60000, 28, 28), (60000,), (10000, 28, 28), (10000,): [0 1 2 3 4 5 6 7 8 9]

enc = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
y_train = enc.fit_transform(y_train).toarray() 
y_test = enc.transform(y_test).toarray()

model = Sequential()
model.add(Dense(units=32, input_shape=(28, 28), activation='relu'))  
model.add(Dropout(0.15))
model.add(Dense(32, activation='relu'))                                    
model.add(MaxPooling1D())                                                              
model.add(Dense(16,))   
model.add(Dropout(0.15))
model.add(Dense(16,))
model.add(MaxPooling1D())                                                       
model.add(Flatten())                                                               
model.add(Dense(64, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))

'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 28, 32)            928
_________________________________________________________________
dropout (Dropout)            (None, 28, 32)            0
_________________________________________________________________
dense_1 (Dense)              (None, 28, 32)            1056
_________________________________________________________________
max_pooling1d (MaxPooling1D) (None, 14, 32)            0
_________________________________________________________________
dense_2 (Dense)              (None, 14, 16)            528
_________________________________________________________________
dropout_1 (Dropout)          (None, 14, 16)            0
_________________________________________________________________
dense_3 (Dense)              (None, 14, 16)            272
_________________________________________________________________
max_pooling1d_1 (MaxPooling1 (None, 7, 16)             0
_________________________________________________________________
flatten (Flatten)            (None, 112)               0
_________________________________________________________________
dense_4 (Dense)              (None, 64)                7232
_________________________________________________________________
dense_5 (Dense)              (None, 16)                1040
_________________________________________________________________
dense_6 (Dense)              (None, 10)                170
=================================================================
Total params: 11,226
Trainable params: 11,226
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
4D DNN w/o GlobalAveragePooling
 > entropy: 0.19406241178512573 accuracy: 0.9405999779701233
* 3D DNN w/o GlobalAveragePooling: 1 minute and 51 seconds
 > entropy: 0.06997454166412354 accuracy: 0.9807000160217285
CNN
 > entropy: 0.05236193165183067 accuracy: 0.9915000200271606)
'''