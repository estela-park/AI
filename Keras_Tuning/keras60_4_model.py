# check if augmenting data solves overfitting and enhances accuracy  

import time
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import QuantileTransformer, OneHotEncoder

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest',
)

test_datagen = ImageDataGenerator(rescale=1./255)

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data() 
# (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28*28) 
# (60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28*28) 
# (10000, 28, 28, 1)

scaler = QuantileTransformer()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

enc = OneHotEncoder()
y_train = enc.fit_transform(y_train).toarray()
y_test = enc.transform(y_test).toarray()

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu' ,input_shape=(28, 28, 1))) 
model.add(Dropout(0.1))    
model.add(Flatten())                                                                              
model.add(Dense(256, activation='relu'))
model.add(Dense(84, activation='selu'))
model.add(Dense(10, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
es = EarlyStopping(monitor='val_loss', patience=8, mode='min', verbose=2, restore_best_weights=True)
start = time.time()
model.fit(train_datagen.flow(x_train, y_train, shuffle=True), epochs=240, batch_size=32, verbose=2, 
          validation_data=test_datagen.flow(x_test, y_test), callbacks=[es])
end = time.time() - start

# when model.evaluate takes x_test, y_test it's not normalized by /255.
# validation_data=datagen.flow() does its job
print('it took',end//1,'seconds')

# Ultimatum: it took 4 mins and 27 secs, acc: 0.5315, val_acc: 0.6812


# Vanilla,        it took 3 minutes 31 seconds with loss: 0.3533134460449219 accuracy: 0.8723999857902527
# w/o MaxPooling, froze at acc=0.1
# w/h MaxPooling, froze at acc=0.1
# GAP -> Flatten, not working
# 2 Conv -> 1,    it took 6 mins, acc: 0.4610, val_acc: 0.5300
#                 there is no sign of overfitting, val_acc(validation_data is intact) >= acc
#                 there is no sign of enhanced accuracy either
# Drop: .25->.1,  it took 8 mins and 20 secs, acc: 0.5208, val_acc: 0.6730
# no Pooling,     it took 9 mins and 35 secs, acc: 0.5623, val_acc: 0.7201
# slimer FC,      it took 12 mins and 1 sec, acc: 0.4760, val_acc: 0.6165


'''
conv2d (Conv2D)              (None, 28, 28, 32)        160
_________________________________________________________________
dropout (Dropout)            (None, 28, 28, 32)        0
_________________________________________________________________
flatten (Flatten)            (None, 25088)             0
_________________________________________________________________
dense (Dense)                (None, 256)               6422784
_________________________________________________________________
dense_1 (Dense)              (None, 84)                21588
_________________________________________________________________
dense_2 (Dense)              (None, 10)                850
'''