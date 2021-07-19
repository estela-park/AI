from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, MaxPool2D, Dropout, GlobalAveragePooling2D, Layer, Input
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
import time

# 1. data-set
(x_train, y_train), (x_test, y_test) = cifar10.load_data() 
# x: (50000, 32, 32, 3) (10000, 32, 32, 3)
# y:  (50000, 1) (10000, 1)

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
output_1 = Dense(10, activation='softmax')(hl1) # KerasTensor (None, 10)

hl2 = GlobalAveragePooling2D()(hl)
hl2 = Dense(10, activation='softmax')(hl2)
output_2 = Layer()(hl2)                         # KerasTensor (None, 10)

# 3. compilation & training
model = Model(inputs=[input_l], outputs=[output_1, output_2])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=12, mode='min', verbose=1)

start = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=64, verbose=2, validation_split=0.15, callbacks=[es])

end = time.time() - start

# 4. evaluation

loss = model.evaluate(x_test, y_test)
print('it took', end/60, 'minutes and', end%60,'seconds')
print('global_ entropy:', loss[0])
print('entropy:', loss[1],'accuracy:', loss[2])
print('entropy:', loss[3],'accuracy:', loss[4])

'''
DNN w/o GlobalAveragePooling   [epochs=44, batch_size=64] *stopped early
    entropy: 1.3354147672653198 accuracy: 0.5220000147819519
DNN w/h GlobalAveragePooling   [epochs=44, batch_size=64] *stopped early
    entropy: 1.7635047435760498 accuracy: 0.36239999532699585
CNN *stopped early             [epochs=42, batch_size=64] 
    entropy: 1.5305525064468384 accuracy: 0.7447999715805054
'''