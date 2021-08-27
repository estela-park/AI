import time
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16
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

vgg16 = VGG16(weights='imagenet', include_top=False,
              input_shape=(32, 32, 3))
# vgg16.trainable = False

model = Sequential()
model.add(vgg16)
model.add(GlobalAveragePooling2D())
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(32))
model.add(Dense(10, activation='softmax'))

model.summary()

start = time.time()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=15, mode='min', verbose=2, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=300, batch_size=512, verbose=2, validation_split=0.25, callbacks=[es])
end = time.time() - start
print('it took', end/60,'minutes and', end%60, 'seconds')

loss = model.evaluate(x_test, y_test)
print('entropy :', loss[0], ', accuracy :', loss[1])


'''
with ReduceLearningRate, stopped early at 161th epo
it took 3 minutes and 27 seconds
entropy : 1.1582015752792358 , accuracy : 0.5856999754905701

with DNN attached to flatten VGG16, trainable, stopped early at 26th epo
it took 3 minutes and 36 seconds
entropy : 0.7677522897720337 , accuracy : 0.7533000111579895

with DNN attached to flatten VGG16, un-trainable, stopped early at 38th epo
it took 2 minutes and 1 seconds
entropy : 1.2070914506912231 , accuracy : 0.582099974155426

with DNN attached to average-pooled VGG16, trainable, stopped early at 26th epo
it took 3 minutes and 37 seconds
entropy : 2.302769899368286 , accuracy : 0.10000000149011612

with DNN attached to average-pooled VGG16, un-trainable, stopped early at 39th epo
it took 2 minutes and 4 seconds
entropy : 1.212439775466919 , accuracy : 0.5853999853134155
'''