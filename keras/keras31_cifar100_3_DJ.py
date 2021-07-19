from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Input
from tensorflow.keras.layers import Layer
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, PowerTransformer, QuantileTransformer, OneHotEncoder
import time
import numpy as np
import matplotlib.pyplot as plt

# 1. data set-up
(x_train, y_train), (x_test, y_test) = cifar100.load_data() # (50000, 32, 32, 3) (10000, 32, 32, 3) (50000, 1) (10000, 1)

start = time.time()

x_train = x_train.reshape(50000, 32 * 32 * 3)
x_test = x_test.reshape(10000, 32 * 32 * 3)

standard_scaler = StandardScaler()
standard_scaler.fit(x_train)
x_train_std = standard_scaler.transform(x_train)
x_test_std = standard_scaler.transform(x_test)

min_max_scaler = MinMaxScaler()
min_max_scaler.fit(x_train)
x_train_mm = min_max_scaler.transform(x_train)
x_test_mm = min_max_scaler.transform(x_test)

max_abs_scaler = MaxAbsScaler()
max_abs_scaler.fit(x_train)
x_train_ma = max_abs_scaler.transform(x_train)
x_test_ma = max_abs_scaler.transform(x_test)

x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)

x_train_mm = x_train_mm.reshape(50000, 32, 32, 3)
x_test_mm = x_test_mm.reshape(10000, 32, 32, 3)

x_train_ma = x_train_ma.reshape(50000, 32, 32, 3)
x_test_ma = x_test_ma.reshape(10000, 32, 32, 3)

enc = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
y_train = enc.fit_transform(y_train).toarray()
y_test = enc.transform(y_test).toarray()   # (10000, 100)

# 2. modelling
input_l = Input(shape=(32, 32, 3))
hl = Conv2D(filters=32, kernel_size=(2, 2), padding='valid', activation='relu', input_shape=(32, 32, 3))(input_l)
hl = Conv2D(32, (2, 2), padding='same', activation='relu')(hl)
hl = MaxPool2D()(hl)
hl = Conv2D(32, (2, 2), padding='valid', activation='relu')(hl)
hl = Conv2D(32, (2, 2), padding='same', activation='relu')(hl)
hl = MaxPool2D()(hl)                         
hl = Conv2D(32, (2, 2), activation='relu')(hl) 
hl = Conv2D(32, (2, 2), padding='same', activation='relu')(hl)
hl = MaxPool2D()(hl)   
hl = Flatten()(hl)
hl = Dense(128, activation='relu')(hl)
hl = Dense(128, activation='relu')(hl)
hl = Dense(128, activation='relu')(hl)
hl = Dense(100, activation='softmax')(hl)
output_l = Layer()(hl)

model = Model(inputs=[input_l], outputs=[output_l])
model_std = Model(inputs=[input_l], outputs=[output_l])
model_mm = Model(inputs=[input_l], outputs=[output_l])
model_ma = Model(inputs=[input_l], outputs=[output_l])

# 3. compilation & training
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model_std.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model_ma.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model_mm.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='acc', patience=8, mode='max', verbose=1)

hist = model.fit(x_train, y_train, epochs=100, batch_size=128, verbose=2, validation_split=0.15, callbacks=[es])
hist_std = model_std.fit(x_train_std, y_train, epochs=100, batch_size=128, verbose=2, validation_split=0.15, callbacks=[es])
hist_mm = model_mm.fit(x_train_mm, y_train, epochs=100, batch_size=128, verbose=2, validation_split=0.15, callbacks=[es])
hist_ma = model_ma.fit(x_train_ma, y_train, epochs=100, batch_size=128, verbose=2, validation_split=0.15, callbacks=[es])

end = time.time() - start

print('it took', end/60,'minutes and', end%60, 'seconds')
# 4. prediction & evaluation

loss = model.evaluate(x_test, y_test)
print('entropy :', loss[0], ', accuracy :', loss[1])

loss_std = model_std.evaluate(x_test_std, y_test)
print('********* with standard scaler ***********')
print('entropy :', loss_std[0], ', accuracy :', loss_std[1])

loss_mm = model_mm.evaluate(x_test_mm, y_test)
print('********* with minmax scaler ***********')
print('entropy :', loss_mm[0], ', accuracy :', loss_mm[1])

loss_ma = model_std.evaluate(x_test_ma, y_test)
print('********* with minabs scaler ***********')
print('entropy :', loss_ma[0], ', accuracy :', loss_ma[1])

plt.figure(figsize=(9, 5))

plt.subplot(2, 1, 1)
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='upper right')

plt.subplot(2, 1, 2)
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.grid()
plt.title('acc')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.legend(['acc', 'val_acc'])

# plt.show

'''
Epoch 00062: early stopping
it took 3.0128586093584695 minutes and 0.7715165615081787 seconds
313/313 [==============================] - 1s 4ms/step - loss: 7.2726 - accuracy: 0.3343
loss[entropy] : 7.272585391998291 , loss[accuracy] : 0.3343000113964081
-
it took 17.48889076312383 minutes and 29.33344578742981 seconds
313/313 [==============================] - 4s 11ms/step - loss: 468.0805 - acc: 0.2845
entropy : 468.08050537109375 , accuracy : 0.28450000286102295
313/313 [==============================] - 4s 14ms/step - loss: 7.5431 - acc: 0.3543
********* with standard scaler ***********
entropy : 7.543056964874268 , accuracy : 0.35429999232292175
'''