import time
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
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
vgg16.trainable = False

model = Sequential()
model.add(vgg16)
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(512))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(100, activation='softmax'))

model.summary()

start = time.time()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=8, mode='min', verbose=2, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=300, batch_size=512, verbose=0, validation_split=0.25, callbacks=[es])
end = time.time() - start
print('it took', end/60,'minutes and', end%60, 'seconds')

loss = model.evaluate(x_test, y_test)
print('entropy :', loss[0], ', accuracy :', loss[1])


'''
with CNN stopped early at epochs=27
entropy: 4.211299896240234 accuracy: 0.32280001044273376

with DNN attached to flatten VGG16, trainable, stopped early at 9th epo
it took 1 minutes and 18 seconds
entropy : 4.6054301261901855 , accuracy : 0.009999999776482582

with DNN attached to flatten VGG16, un-trainable, stopped early at 16th epo
it took 53 seconds
entropy : 2.645085334777832 , accuracy : 0.3407000005245209

with DNN attached to average-pooled VGG16, trainable, stopped early at 23th epo
it took 3 minutes and 10 seconds
entropy : 2.670217514038086 , accuracy : 0.33820000290870667

with DNN attached to average-pooled VGG16, un-trainable, stopped early at 15th epo
it took 53 seconds
entropy : 2.6464948654174805 , accuracy : 0.34389999508857727
'''