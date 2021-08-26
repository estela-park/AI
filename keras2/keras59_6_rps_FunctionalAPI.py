import numpy as np
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Dropout, MaxPool2D, Dense, Input, BatchNormalization, GlobalAveragePooling2D, Activation, Flatten
from tensorflow.keras.callbacks import EarlyStopping


# 1. Data Preperation
'''
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

xy_train = train_datagen.flow_from_directory(
    '../_data/rps',
    target_size=(150, 150),
    batch_size=2520,
    class_mode='categorical',
    shuffle=True
)
# Found 2520 images belonging to 3 classes.

np.save('../_save/_npy/k59_6_train_x.npy', arr=xy_train[0][0])
np.save('../_save/_npy/k59_6_train_y.npy', arr=xy_train[0][1])
'''

x_train = np.load('../_save/_npy/k59_rps_x_train.npy')
x_test = np.load('../_save/_npy/k59_rps_x_test.npy')
y_train = np.load('../_save/_npy/k59_rps_y_train.npy')
y_test = np.load('../_save/_npy/k59_rps_y_test.npy')

# 2. Modelling
inputL = Input(shape=(150, 150, 3))
hl = Conv2D(filters=40, kernel_size=(3, 3), activation='relu')(inputL)
hl = Conv2D(40, (3, 3), activation='relu')(hl)
hl = MaxPool2D((2, 2))(hl)
hl = Conv2D(filters=20, kernel_size=(2, 2), activation='relu')(hl)
hl = Conv2D(20, (2, 2), activation='relu')(hl)
hl = MaxPool2D((2, 2))(hl)
# hl = BatchNormalization()(hl)
hl = Conv2D(filters = 32, kernel_size=(3,3), activation= 'relu')(hl)
hl = Flatten()(hl)
# hl = Dense(256, activation='elu')(hl)
hl = Dense(128, activation='relu')(hl)
# hl = Dense(32, activation='elu')(hl)
hl = Dense(64, activation='relu')(hl)
hl = Dense(3, activation='relu')(hl)
outputL = Activation('softmax')(hl)
# outputL = Layer()(hl): doesn't change a  thing.

model = Model(inputs=inputL, outputs=outputL)

model.summary()



# 3. compilation & training
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_acc', patience=8, mode='max', verbose=1)

start = time.time()

hist = model.fit(x_train, y_train, epochs=100, validation_split=0.15, batch_size=32, verbose=2, callbacks=[es])
end = time.time() - start

print('it took', end/60,'minutes and', end%60, 'seconds')



# 4. prediction & evaluation
print('entropy:', hist.history['loss'], ', accuracy :', hist.history['acc'])
print('val_entropy:', hist.history['val_loss'], 'val_accuracy :',hist.history['val_acc'])
print(type(hist.history['loss']))
print(len(hist.history['loss']))

'''
it took 1.389906930923462 minutes and 23.394415855407715 seconds
entropy: [1.6855745315551758, ... 0.47765639424324036]
val_entropy: [4.33238410949707, ... 0.33068782091140747]
<class 'list'>
28

it took 0.5264794905980428 minutes and 31.58876943588257 seconds
entropy: [1.364863634109497, ... 1.0581257343292236] , accuracy : [0.34423828125, ... 0.4424031674861908]     
val_entropy: [1.3201783895492554, ... 0.3253968358039856]
<class 'list'>
9

--Flatten -> GlobalAvg
it took 0.8354286273320516 minutes and 50.125717639923096 seconds
entropy: [1.0983033180236816, ... 1.0926249027252197] , accuracy : [0.3330078125, ... 0.37437933683395386]
val_entropy: [1.1108559370040894, ... 1.1251918077468872] val_accuracy : [0.32010582089424133, ... 0.32804232835769653]
<class 'list'>
16

in the Dense layer's units, consider using something in the range [128, 512].
it took 1.1501685420672099 minutes and 9.010112524032593 seconds
entropy: [1.1007471084594727, 1.0986841917037964, 1.098664402961731, 1.0986592769622803, 1.0986909866333008, 1.098670482635498, 
1.098678708076477, 1.0986497402191162, 1.0986899137496948] , accuracy : [0.3272642493247986, 0.3352007567882538, 0.3352007567882538, 0.3352007567882538, 0.3352007567882538, 0.3165266215801239, 0.3225957155227661, 0.3352007567882538, 0.3352007567882538]    
val_entropy: [1.0982853174209595, 1.0986827611923218, 1.0987437963485718, 1.0987634658813477, 1.0988808870315552, 1.0987659692764282, 1.0988545417785645, 1.0988788604736328, 1.098862886428833] val_accuracy : [0.36243385076522827, 0.32275131344795227, 0.32275131344795227, 0.32275131344795227, 0.32275131344795227, 0.3253968358039856, 0.32275131344795227, 0.32275131344795227, 0.32275131344795227]
<class 'list'>
9

Adding Activation layer separately
<class 'list'>
17

width adjusted > no use

MaxPool arg: (2, 2) > no use

made it exact > no use

kernel adjusted > no use

data loaded from other generator > worked
'''