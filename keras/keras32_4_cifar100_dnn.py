import time
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, MaxPool2D, Input, GlobalAveragePooling2D, Dropout, Layer
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler



# 1. data-set
(x_train, y_train), (x_test, y_test) = cifar100.load_data() 
# x: (50000, 32, 32, 3) (10000, 32, 32, 3)
# y: (50000, 1) (10000, 1)

x_train = x_train.reshape(50000, 32 * 32 * 3)
x_test = x_test.reshape(10000, 32 * 32 * 3)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)

enc = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
y_train = enc.fit_transform(y_train).toarray()
y_test = enc.transform(y_test).toarray()

# 2. modeling

input_l = Input(shape=(32, 32, 3))
hl = Dense(32, activation='relu')(input_l)
hl = Dropout(0.2)(hl)
hl = Dense(64, activation='relu')(hl)
hl = MaxPool2D()(hl)
hl = Dense(32, activation='relu')(hl)
hl = Dropout(0.2)(hl)
hl = Dense(64, activation='relu')(hl)
hl = MaxPool2D()(hl)                         
hl = Dense(32, activation='relu')(hl)
hl = Dropout(0.2)(hl)
hl = Dense(64, activation='relu')(hl)
hl = MaxPool2D()(hl)

hl1 = Flatten()(hl)
output_1 = Dense(100, activation='softmax')(hl1) # KerasTensor (None, 10)

hl2 = GlobalAveragePooling2D()(hl)
hl2 = Dense(100, activation='softmax')(hl2)
output_2 = Layer()(hl2)                         # KerasTensor (None, 10)

# 3. compilation & training
model = Model(inputs=[input_l], outputs=[output_1, output_2])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=16, mode='min', verbose=1)

start = time.time()
hist = model.fit(x_train, y_train, epochs=200, batch_size=64, verbose=2, validation_split=0.15, callbacks=[es])

end = time.time() - start

# 4. evaluation

loss = model.evaluate(x_test, y_test)
print('it took', end/60, 'minutes and', end%60,'seconds')
print('global_ entropy:', loss[0])
print('entropy:', loss[1],'accuracy:', loss[3])
print('entropy:', loss[2],'accuracy:', loss[4])

'''
DNN w/o GlobalAveragePooling   [epochs=100, batch_size=32]
    entropy: 3.238474130630493 accuracy: 0.22609999775886536
DNN w/h GlobalAveragePooling   [epochs=100, batch_size=32]
    entropy: 3.914315938949585 accuracy: 0.1014999970793724
CNN *stopped early             [epochs=27, batch_size=64] 
    entropy: 4.211299896240234 accuracy: 0.32280001044273376
'''