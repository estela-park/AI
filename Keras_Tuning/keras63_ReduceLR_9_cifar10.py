from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Input, Dropout, GlobalAveragePooling2D
from tensorflow.keras.layers import Layer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import time

# 1. data set-up
(x_train, y_train), (x_test, y_test) = cifar10.load_data() 
# (50000, 32, 32, 3) (10000, 32, 32, 3) (50000, 1) (10000, 1)

x_train = x_train.reshape(50000, 32 * 32 * 3)
x_test = x_test.reshape(10000, 32 * 32 * 3)

min_max_scaler = MinMaxScaler()
min_max_scaler.fit(x_train)
x_train_mm = min_max_scaler.transform(x_train)
x_test_mm = min_max_scaler.transform(x_test)

x_train_mm = x_train_mm.reshape(50000, 32, 32, 3)
x_test_mm = x_test_mm.reshape(10000, 32, 32, 3)

enc = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
y_train = enc.fit_transform(y_train).toarray()
y_test = enc.transform(y_test).toarray()   
# (10000, 100)


# 2. modelling
input_l = Input(shape=(32, 32, 3))
hl = Conv2D(filters=16, kernel_size=(2, 2), padding='valid', activation='relu', input_shape=(32, 32, 3))(input_l)
hl = Dropout(0.2)(hl)
hl = Conv2D(32, (2, 2), padding='same', activation='relu')(hl)
hl = MaxPool2D()(hl)
hl = Dropout(0.2)(hl)                       
hl = Conv2D(64, (2, 2), activation='relu')(hl) 
hl = Dropout(0.2)(hl)
hl = Conv2D(32, (2, 2), padding='same', activation='relu')(hl)
hl = MaxPool2D()(hl)
hl = GlobalAveragePooling2D()(hl)
hl = Dense(10, activation='softmax')(hl)
output_l = Layer()(hl)

model_mm = Model(inputs=[input_l], outputs=[output_l])


# 3. compilation & training
start = time.time()
model_mm.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=15, mode='min', verbose=2, restore_best_weights=True)
lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto', verbose=2, factor=0.5)
hist_mm = model_mm.fit(x_train_mm, y_train, epochs=300, batch_size=512, verbose=2, validation_split=0.25, callbacks=[es, lr])
end = time.time() - start
print('it took', end/60,'minutes and', end%60, 'seconds')


# 4. prediction & evaluation
loss_mm = model_mm.evaluate(x_test_mm, y_test)
print('entropy :', loss_mm[0], ', accuracy :', loss_mm[1])

'''
+stopped early at 161th epo
it took 3 minutes and 27 seconds
entropy : 1.1582015752792358 , accuracy : 0.5856999754905701
'''