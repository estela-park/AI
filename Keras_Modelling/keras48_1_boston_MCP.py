import numpy as np
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Conv1D, Dropout, GlobalAveragePooling1D, MaxPool1D
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler

datasets = load_boston()

x = datasets.data 
# (506, 13)
y = datasets.target 
# (506,)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, shuffle=True, random_state=78)
# (404, 13) (102, 13)

scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

model1 = Sequential()
model1.add(Conv1D(filters=32, kernel_size=2, padding='same', activation='relu', input_shape=(13, 1))) 
model1.add(Dropout(0.25))
model1.add(Conv1D(32, 2, padding='same', activation='relu'))
model1.add(Dropout(0.2))
model1.add(MaxPool1D())
model1.add(Conv1D(128, 2, padding='same', activation='relu'))
model1.add(MaxPool1D())
model1.add(GlobalAveragePooling1D())
model1.add(Dense(1))

model1.summary()

es = EarlyStopping(monitor='val_loss', patience=24, mode='auto', verbose=2, restore_best_weights=False)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath='./_save/modelcheckpoint/keras48_boston_mcp.hdf5')

start = time.time()
model1.fit(x_train, y_train, epochs=300, batch_size=32, verbose=2, validation_split=0.15, callbacks=[es])
end = time.time() - start

y_predict = model1.predict([x_test])

model1.compile(loss='mse', optimizer='adam')
model1.fit(x_train, y, batch_size=8, epochs=160, verbose=0, validation_split=0.15, callbacks=[es, mcp])
model1.save('./_save/keras48_boston_save_model.hd')

print('=========================Default==========================')
loss = model1.evaluate(x_test, y_test)
prediction  = model1.predict(x_test)
r2 = r2_score(y_test, prediction)
print('loss:', loss)
print('R2 score:', r2)

print('=========================load_model==========================')

model2 = load_model('./_save/keras48_boston_save_model.hd')
loss = model2.evaluate(x_test, y_test)
prediction  = model2.predict(x_test)
r2 = r2_score(y_test, prediction)
print('loss:', loss)
print('R2 score:', r2)

print('=========================MCPwhBEST==========================')

model3 = load_model('./_save/keras49_save_model.hd')
model3.load_weights('./_save/modelcheckpoint/keras48_boston_mcp.hdf5')
loss = model3.evaluate(x_test, y_test)
prediction  = model3.predict(x_test)
r2 = r2_score(y_test, prediction)
print('loss:', loss)
print('R2 score:', r2)

'''
*****************SAVING_trained model==ES********************
=========================Default=============================
loss: 3914.936767578125
R2 score: -2.9623796701759457
=========================load_model==========================
loss: 3914.936767578125
R2 score: -2.9623796701759457
=========================MCPwhBEST===========================
loss: 7298.0927734375
R2 score: -6.386534744685374


********************loading model, weights*******************
=========================Default=============================
loss: 2401.33740234375
R2 score: -1.4304381360608485
=========================load_model==========================
loss: 3914.936767578125
R2 score: -2.9623796701759457
=========================MCPwhBEST===========================
loss: 7298.0927734375
R2 score: -6.386534744685374


*************ES's option restore_best_weights=True***********
=========================Default=============================
loss: 1146.809814453125
R2 score: -0.16070735189040275
=========================load_model==========================
loss: 1146.809814453125
R2 score: -0.16070735189040275
=========================MCPwhBEST===========================
loss: 1146.809814453125
R2 score: -0.16070735189040275
'''