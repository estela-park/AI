import time
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Input, Layer
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler, OneHotEncoder

x_data = np.load('../_save/_npy/k55_x_data_cifar100.npy')
y_data = np.load('../_save/_npy/k55_y_data_cifar100.npy')

(x_train, y_train), (x_test, y_test) = (x_data[:50000], y_data[:50000]), (x_data[50000:], y_data[50000:])

x_train = x_train.reshape(50000, 32 * 32 * 3)
x_test = x_test.reshape(10000, 32 * 32 * 3)
standard_scaler = StandardScaler()
x_train_std = standard_scaler.fit_transform(x_train)
x_test_std = standard_scaler.transform(x_test)
x_train_std = x_train_std.reshape(50000, 32, 32, 3)
x_test_std = x_test_std.reshape(10000, 32, 32, 3)

enc = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
y_train = enc.fit_transform(y_train).toarray()
y_test = enc.transform(y_test).toarray()

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

start = time.time()
model_std = Model(inputs=[input_l], outputs=[output_l])
model_std.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_acc', patience=24, mode='max', verbose=1, restore_best_weights=True)
hist_std = model_std.fit(x_train_std, y_train, epochs=100, batch_size=128, verbose=2, validation_split=0.15, callbacks=[es])
end = time.time() - start

loss_std = model_std.evaluate(x_test_std, y_test)
print('it took', end//60,'minutes and', end%60//1, 'seconds, entropy :', loss_std[0], ', accuracy :', loss_std[1])

# it took 2 minutes and 17 seconds, entropy: 2.947, accuracy: 0.3569
# DNN w/o GAP                       entropy: 3.238, accuracy: 0.2261
# DNN w/h GAP                       entropy: 3.914, accuracy: 0.1015
# CNN                               entropy: 4.211, accuracy: 0.3228
# Conv1D                            entropy: 2.832, accuracy: 0.3026
# LSTM                              entropy: 4.247, accuracy: 0.0528