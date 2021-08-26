from tensorflow.keras.datasets import fashion_mnist
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping
import time


# 1. data-set

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data() 
# x: (60000, 28, 28) (60000,) 
# y: (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28 * 28 * 1)
x_test = x_test.reshape(10000, 28 * 28 * 1)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

enc = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
y_train = enc.fit_transform(y_train).toarray()
y_test = enc.transform(y_test).toarray()

# 2. modeling

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu' ,input_shape=(28, 28, 1))) 
model.add(Conv2D(32, (5, 5), padding='same', activation='relu'))                   
model.add(MaxPool2D())                                         
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))                   
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))    
model.add(Flatten())                                              
model.add(Dense(256, activation='relu'))
model.add(Dense(124, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 3. compilation & training
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=2)

start = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=128, verbose=2, validation_split=0.15, callbacks=[es])
end = time.time() - start

# 4. evaluation

loss = model.evaluate(x_test, y_test)
print('it took', end/60, 'minutes and', end%60,'seconds')
print('entropy:', loss[0],'accuracy:', loss[1])

'''
epochs=24, batch_size=128, stopped early
it took 1 minute and 13 seconds
entropy: 0.6042917966842651 accuracy: 0.9204000234603882
'''