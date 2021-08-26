# pre-compilation n training model doesn't have set weitght in it
# post-compilation n training model has set weitght in it

import time
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Input, Dropout, Layer
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


# 1. data set-up
(x_train, y_train), (x_test, y_test) = mnist.load_data() 
# (50000, 28, 28, 1) (10000, 28, 28, 1) (50000, 1) (10000, 1)

x_train = x_train.reshape(60000, 28 * 28 * 1)
x_test = x_test.reshape(10000, 28 * 28 * 1)

min_max_scaler = MinMaxScaler()
x_train_mm = min_max_scaler.fit_transform(x_train).reshape(60000, 28, 28, 1)
x_test_mm = min_max_scaler.transform(x_test).reshape(10000, 28, 28, 1)

enc = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
y_train = enc.fit_transform(y_train).toarray()
y_test = enc.transform(y_test).toarray() 


# 2. modelling
input_l = Input(shape=(28, 28, 1))
hl = Conv2D(filters=32, kernel_size=(2, 2), padding='valid', activation='relu', input_shape=(32, 32, 3))(input_l)
hl = Dropout(0.2)(hl)
hl = Conv2D(32, (2, 2), padding='same', activation='relu')(hl)
hl = MaxPool2D()(hl)                         
hl = Conv2D(64, (2, 2), activation='relu')(hl) 
hl = Dropout(0.2)(hl)
hl = MaxPool2D()(hl)   
hl = Flatten()(hl)
hl = Dense(128, activation='relu')(hl)
hl = Dropout(0.2)(hl)
hl = Dense(128, activation='relu')(hl)
hl = Dropout(0.2)(hl)
hl = Dense(10, activation='softmax')(hl)
output_l = Layer()(hl)

model_mm = Model(inputs=[input_l], outputs=[output_l])
model_mm.save('../_save/_keras/keras45_1.h5')


# 3. compilation & training
start = time.time()
model_mm.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='acc', patience=8, mode='max', verbose=1, restore_best_weights=True)
hist_mm = model_mm.fit(x_train_mm, y_train, epochs=100, batch_size=32, verbose=2, validation_split=0.25, callbacks=[es])
end = time.time() - start

print('it took', end//60,'minutes and', end%60//1, 'seconds')


# 4. prediction & evaluation
loss_mm = model_mm.evaluate(x_test_mm, y_test)
print('********* with minmax scaler ***********')
print('entropy :', loss_mm[0], ', accuracy :', loss_mm[1])

'''
*** with minmax scaler, it took 6.0 minutes and 22.0 seconds
entropy : 0.03099539503455162 , accuracy : 0.9933000206947327
'''