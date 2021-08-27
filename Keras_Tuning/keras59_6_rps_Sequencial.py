import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping


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
    validation_split=0.25
    # ^ difference1
)

test_datagen = ImageDataGenerator(rescale=1./255)

xy_train = train_datagen.flow_from_directory(
    '../_data/rps',
    target_size=(150, 150),
    batch_size=2000,
    class_mode='categorical',
    shuffle=True,
    subset='training'
    # ^ difference2
)
# Found 1890 images belonging to 3 classes.


xy_test = train_datagen.flow_from_directory(
    '../_data/rps',
    target_size=(150, 150),
    batch_size=2000,
    class_mode='categorical',
    shuffle=True,
    subset='validation'
    # ^ difference3
)
# Found 630 images belonging to 3 classes.

np.save('../_save/_npy/k59_rps_x_train', arr=xy_train[0][0])
np.save('../_save/_npy/k59_rps_y_train', arr=xy_train[0][1])
np.save('../_save/_npy/k59_rps_x_test', arr=xy_test[0][0])
np.save('../_save/_npy/k59_rps_y_test', arr=xy_test[0][1])


# xy_train[0][0].shape: (1890, 150, 150, 3)
# xy_train[0][1].shape: (1890, 3)

x_train = np.load('../_save/_npy/k59_rps_x_train.npy')
x_test = np.load('../_save/_npy/k59_rps_x_test.npy')
y_train = np.load('../_save/_npy/k59_rps_y_train.npy')
y_test = np.load('../_save/_npy/k59_rps_y_test.npy')


model = Sequential()
# ^ difference4: Functional > Sequential

model.add(Conv2D(32, kernel_size=(2, 2), input_shape=(150, 150, 3)))
model.add(Conv2D(32, (2, 2), activation= 'relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(16, kernel_size=(2,2), activation= 'relu'))
model.add(Conv2D(16, kernel_size=(2,2), activation= 'relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
# model.add(Dense(256, activation= 'relu'))
model.add(Dense(128, activation= 'relu'))
# model.add(Dense(32, activation= 'relu'))
model.add(Dense(64, activation= 'relu'))
model.add(Dense(3, activation= 'softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])


es = EarlyStopping(monitor='val_acc', patience=10, mode='auto', verbose=2)

hist = model.fit(x_train, y_train, epochs=500, callbacks=[es], validation_split=0.1,
                steps_per_epoch=32, validation_steps=1, verbose=2)

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']


loss = model.evaluate(x_test, y_test)

print('entropy:', hist.history['loss'], ', accuracy :', hist.history['acc'])
print('val_entropy:', hist.history['val_loss'], 'val_accuracy :',hist.history['val_acc'])
print('loss : ',loss)

'''
prints multiple losses and accuracy, 
the only significant change is that it used Sequencial.
but unlike previous Functional API it didn't stuck at loss/acc.

changed model's layers to match previous Functional model
  > that is adding Dropout after each Conv layer and switching Flatten -> GlobalAVG
  > model froze

removed all the Dropout
  > no changes

exchange GAP to Flatten
  > no changes

MaxPool arg: (2, 2)
  > no changes

the # of nodes, filters, units adjusted
  > worked
  > ***the complexity should match that of desired outcome***

give validation_split=# args for ImageDataGenerator
then subset='train' or 'validation' for flow_from_directory

Still not quiet getting why the result gives multiple losses and accuracies.
Perhaps it's for layers' multiplicity.
'''