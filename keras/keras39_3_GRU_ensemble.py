import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GRU, Input, concatenate
from tensorflow.keras.callbacks import EarlyStopping
import time

x1 = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12],[20,30,40],[30,40,50],[40,50,60]])
x2 = np.array([[10,20,30],[20,30,40],[30,40,50],[40,50,60],[50,60,70],[60,70,80],[70,80,90],[80,90,100],[90,100,110],[100,110,120],[2,3,4],[3,4,5],[4,5,6]])
y  = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x1_predict = np.array([[55],[65],[75]]).reshape(1,3,1)
x2_predict = np.array([[65],[75],[85]]).reshape(1,3,1)

x1 = x1.reshape(x1.shape[0], x1.shape[1], 1)
x2 = x2.reshape(x2.shape[0], x2.shape[1], 1)

inL_l = Input(shape=(3,1))
inL_r = Input(shape=(3,1))
hl_l = GRU(units=32, activation='relu')(inL_l)
hl_r = GRU(units=32, activation='relu')(inL_r)
hl_l = Dense(32, activation='relu')(hl_l)
hl_r = Dense(32, activation='relu')(hl_r)
hl_l = Dense(16, activation='relu')(hl_l)
hl_r = Dense(16, activation='relu')(hl_r)
hl_l = Dense(16, activation='relu')(hl_l)
hl_r = Dense(16, activation='relu')(hl_r)
hl_l = Dense(8, activation='relu')(hl_l)
hl_r = Dense(8, activation='relu')(hl_r)
outL_l = Dense(1, activation='relu')(hl_l)
outL_r = Dense(1, activation='relu')(hl_r)
outL = concatenate([outL_l, outL_r])
outL = Dense(10, activation='relu')(outL)
outL = Dense(10, activation='relu')(outL)
outL = Dense(1)(outL)


model = Model(inputs=[inL_l, inL_r], outputs=[outL])
model.summary()

start = time.time()
es = EarlyStopping(monitor='val_loss', patience=24, mode='min', verbose=2)
model.compile(loss='mse', optimizer='adam')
model.fit([x1, x2], y, batch_size=16, epochs=1000, verbose=2, validation_split=0.2, callbacks=[es])
end = time.time() - start

predict = model.predict([x1_predict, x2_predict])

print('time spent:',end,'seconds')
print('prediction:',predict)