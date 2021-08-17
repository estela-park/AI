import time
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# (6000, 28, 28) (10000, 28, 28)

x = np.append(x_train, x_test, axis=0)
y = np.append(y_train, y_test, axis=0)
# (70000, 28, 28)

x = x.reshape(70000, 28*28)

enc = OneHotEncoder()
y = enc.fit_transform(y.reshape(-1, 1)).toarray()

pca = PCA(n_components=196)
x = pca.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, random_state=76)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_train = x_train.reshape(59500, 14, 14, 1)
x_test = scaler.transform(x_test)
x_test = x_test.reshape(10500, 14, 14, 1)

model = Sequential()
model.add(Conv2D(128, (2, 2), input_shape=(14, 14, 1), padding='same', activation='relu'))  
model.add(Dropout(0.2)) 
model.add(Conv2D(128, (2, 2), activation='relu'))  
model.add(Dropout(0.2)) 
model.add(MaxPool2D())
model.add(Flatten())         
model.add(Dense(512, activation='relu'))                                                                                                                                       
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25)) 
model.add(Dense(64, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))

es = EarlyStopping(monitor='val_loss', patience=8, mode='min', verbose=2)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

start = time.time()
model.fit(x_train, y_train, epochs=120, verbose=2, validation_split=0.15, batch_size=256, callbacks=[es])
end = time.time() - start

loss = model.evaluate(x_test, y_test)

print('it took', end//60, 'minutes and', end%60,'seconds')
print('entropy:', loss[0],'accuracy:', loss[1])

'''
CNN without PCA
it took 1 minute and 10 seconds
entropy: 0.05236193165183067 accuracy: 0.9915000200271606

**CNN with PCA: stopped early at 30
it took 1 minute and 5 seconds
entropy: 0.1740589141845703 accuracy: 0.9527618885040283

DNN without PCA: stopped early at 71
it took 1 minute
entropy: 0.10611440986394882 accuracy: 0.9824761748313904

DNN with PCA 95% explanation: stopped early at 49
it took 40 seconds
entropy: 0.17585872113704681 accuracy: 0.948190450668335
'''