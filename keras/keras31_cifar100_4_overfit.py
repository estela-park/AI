# how to overcome overfitting
#
# 1. increase the size of training data(not by reducing valid/test data)
# 2. normalize the data passed from one layer to another
# 3. dropout


import time
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Input, Dropout
from tensorflow.keras.layers import Layer
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, OneHotEncoder


# 1. data set-up
(x_train, y_train), (x_test, y_test) = cifar100.load_data() 
# (50000, 32, 32, 3) (10000, 32, 32, 3) (50000, 1) (10000, 1)

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

x_train_std = x_train_std.reshape(50000, 32, 32, 3)
x_test_std = x_test_std.reshape(10000, 32, 32, 3)

x_train_mm = x_train_mm.reshape(50000, 32, 32, 3)
x_test_mm = x_test_mm.reshape(10000, 32, 32, 3)

x_train_ma = x_train_ma.reshape(50000, 32, 32, 3)
x_test_ma = x_test_ma.reshape(10000, 32, 32, 3)

enc = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
y_train = enc.fit_transform(y_train).toarray()
y_test = enc.transform(y_test).toarray() 

# 2. modelling
input_l = Input(shape=(32, 32, 3))
hl = Conv2D(filters=32, kernel_size=(2, 2), padding='valid', activation='relu', input_shape=(32, 32, 3))(input_l)
hl = Dropout(0.2)(hl)
hl = Conv2D(32, (2, 2), padding='same', activation='relu')(hl)
hl = MaxPool2D()(hl)                    
hl = Flatten()(hl)
hl = Dense(128, activation='relu')(hl)
hl = Dropout(0.2)(hl)
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

es = EarlyStopping(monitor='acc', patience=10, mode='max', verbose=1)

hist = model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=2, validation_split=0.25, callbacks=[es])
hist_std = model_std.fit(x_train_std, y_train, epochs=100, batch_size=32, verbose=2, validation_split=0.25, callbacks=[es])
hist_mm = model_mm.fit(x_train_mm, y_train, epochs=100, batch_size=32, verbose=2, validation_split=0.25, callbacks=[es])
hist_ma = model_ma.fit(x_train_ma, y_train, epochs=100, batch_size=32, verbose=2, validation_split=0.25, callbacks=[es])

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

'''
it took 4.938602916399637 minutes and 56.31617498397827 seconds
313/313 [==============================] - 1s 2ms/step - loss: 4.6055 - acc: 0.0100
entropy : 4.605531692504883 , accuracy : 0.009999999776482582
313/313 [==============================] - 1s 2ms/step - loss: 4.6055 - acc: 0.0100
********* with standard scaler ***********
entropy : 4.605531692504883 , accuracy : 0.009999999776482582
313/313 [==============================] - 1s 2ms/step - loss: 4.6055 - acc: 0.0100
********* with minmax scaler ***********
entropy : 4.605531692504883 , accuracy : 0.009999999776482582
313/313 [==============================] - 1s 2ms/step - loss: 4.6055 - acc: 0.0100
********* with minabs scaler ***********
entropy : 4.605531692504883 , accuracy : 0.009999999776482582
'''