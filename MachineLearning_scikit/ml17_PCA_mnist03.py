# work flow
# > 1. merge the train data and test data
# > 2. reduce the variance with 99.9% of explanatory
# > 3. split the reduced data into train/test
# > 4. scaling & one-hot encoding
# > 5. comparison

import time
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
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

pca = PCA(n_components=486)
x = pca.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, random_state=76)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


model = Sequential()
model.add(Dense(512, input_shape=(486, ), activation='relu'))  
model.add(Dropout(0.15))                                                                                            
model.add(Dense(256, activation='relu'))                  
model.add(Dropout(0.15))                                                                                                
model.add(Dense(128, activation='relu'))                  
model.add(Dropout(0.15))                                      
model.add(Dense(64, activation='relu'))                                                                    
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.25)) 
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

DNN without PCA: stopped early at 71
it took 1 minute
entropy: 0.10611440986394882 accuracy: 0.9824761748313904

DNN with PCA 99.9% explanation: stopped early at 51
   >> Dense layers had too few nodes, widened and deepened the model
it took  48 seconds
entropy: 1.4746172428131104 accuracy: 0.45295238494873047

   >> Depth to match vanilla DNN: stopped early at 49
it took 41 seconds
entropy: 0.5348224639892578 accuracy: 0.8471428751945496

'''