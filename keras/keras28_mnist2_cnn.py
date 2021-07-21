import time
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Flatten
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping

(x_train, y_train), (x_test, y_test) = mnist.load_data() # (60000, 28, 28), (60000,), (10000, 28, 28), (10000,): [0 1 2 3 4 5 6 7 8 9]

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

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5), input_shape=(28, 28, 1), padding='same', activation='relu'))  
model.add(Conv2D(32, (5, 5), activation='relu'))                                    
model.add(MaxPool2D())                                                              
model.add(Conv2D(16, (3, 3)))                                                       
model.add(Conv2D(16, (3, 3)))      
model.add(MaxPool2D())                                                              
model.add(Flatten())                                                               
model.add(Dense(64, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))

es = EarlyStopping(monitor='loss', patience=8, mode='min', verbose=2)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

start = time.time()
model.fit(x_train, y_train, epochs=120, verbose=2, validation_split=0.15, batch_size=256, callbacks=[es])
end = time.time() - start

loss = model.evaluate(x_test, y_test)

print('it took', end/60, 'minutes and', end%60,'seconds')
print('entropy:', loss[0],'accuracy:', loss[1])

'''
epochs=45, batch_size=256 stopped early
it took 1 minute and 10 seconds
entropy: 0.05236193165183067 accuracy: 0.9915000200271606
'''