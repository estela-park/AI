from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPool1D, Input, Dropout, Layer, GlobalAveragePooling1D, LSTM
from tensorflow.python.keras.layers.core import Flatten
import time

# data-set
(x_train, y_train), (x_test, y_test) = mnist.load_data() 
x_train = x_train.reshape(60000, 28*28*1)
x_test = x_test.reshape(10000, 28*28*1)


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 

x_train = x_train.reshape(60000, 28, 28)
x_test = x_test.reshape(10000, 28, 28)

enc = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
y_train = enc.fit_transform(y_train).toarray()
y_test = enc.transform(y_test).toarray() 

# 2. modeling
input_l = Input(shape=(28, 28))
hl = LSTM(32, activation='relu')(input_l)
hl = Dropout(0.2)(hl)
hl = Dense(64, activation='relu')(hl)
hl = Dropout(0.2)(hl)

hl1 = Flatten()(hl)
output_1 = Dense(10, activation='softmax')(hl1) # KerasTensor (None, 10)

hl2 = GlobalAveragePooling1D()(hl)
hl2 = Dense(10, activation='softmax')(hl2)
output_2 = Layer()(hl2)                         # KerasTensor (None, 10)

hlc = Conv1D(filters=32, kernel_size=2, padding='valid', activation='relu')(input_l)
hlc = Dropout(0.2)(hlc)                        
hlc = Conv1D(64, 2, activation='relu')(hlc) 
hlc = Dropout(0.2)(hlc)
hlc = MaxPool1D()(hlc)
hlc = GlobalAveragePooling1D()(hlc)
hlc = Dense(10, activation='softmax')(hlc)
output_c = Layer()(hlc)                         # KerasTensor (None, 10)

# 3. compilation & training
model = Model(inputs=[input_l], outputs=[output_1, output_2, output_c])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=15, mode='min', verbose=1)

start = time.time()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=2, validation_split=0.15, callbacks=[es]) 
end = time.time() - start

# 4. evaluation & prediction
loss = model.evaluate(x_test, y_test)
print('it took', end/60, 'minutes and', end%60,'seconds')
print('entropy:', loss[0],'accuracy:', loss[1])