from numpy import fliplr
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.layers import Layer
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from sklearn.preprocessing import MinMaxScaler
import time
import matplotlib.pyplot as plt

# 1. data set-up
(x_train, y_train), (x_test, y_test) = cifar100.load_data() # (50000, 32, 32, 3) (10000, 32, 32, 3) (50000, 1) (10000, 1)

start = time.time()

x_train = x_train.reshape(50000, 32 * 32 * 3)
x_test = x_test.reshape(10000, 32 * 32 * 3)

min_max_scaler = MinMaxScaler()
min_max_scaler.fit(x_train)
x_train_mm = min_max_scaler.transform(x_train)
x_test_mm = min_max_scaler.transform(x_test)

x_train_mm = x_train_mm.reshape(50000, 32, 32, 3)
x_test_mm = x_test_mm.reshape(10000, 32, 32, 3)

# 2. modelling
input_l = Input(shape=(32, 32, 3))
hl = Dense(128, activation='relu')(hl)
hl = Dense(128, activation='softmax')(hl)
hl = Dense(128, activation='relu')(hl)
hl = flatten()(hl)
hl = Dense(100, activation='softmax')(hl)
output_l = Layer()(hl)

model_mm = Model(inputs=[input_l], outputs=[output_l])

# 3. compilation & training
model_mm.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_acc', patience=16, mode='max', verbose=1)
tb = TensorBoard(log_dir='./_save/_graph', histogram_freq=5, write_graph=True, write_images=True)

hist_mm = model_mm.fit(x_train_mm, y_train, epochs=20, batch_size=128, verbose=2, validation_split=0.15, callbacks=[es, tb])

end = time.time() - start

print('it took', end/60,'minutes and', end%60, 'seconds')
# 4. prediction & evaluation

loss_mm = model_mm.evaluate(x_test_mm, y_test)
print('********* with minmax scaler ***********')
print('entropy :', loss_mm[0], ', accuracy :', loss_mm[1])