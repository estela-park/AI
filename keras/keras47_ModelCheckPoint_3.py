# Bypassing a Trouble
# -overriding saved model with the same name 

import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, concatenate
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score

x1 = np.array([range(100), range(301, 401), range(1, 101)])
x2 = np.array([range(101, 201), range(411, 511), range(100, 200)])
x1 = np.transpose(x1)
x2 = np.transpose(x2)

y1 = np.array([range(1001, 1101)])
y1 = np.transpose(y1)

x1_train, x1_test, x2_train, x2_test, y1_train, y1_test = train_test_split(x1, x2, y1, train_size=0.85, random_state=72)

input11 = Input(shape=(3, ))
dense11 = Dense(48, activation='relu')(input11)
dense12 = Dense(12, activation='relu')(dense11)
dense13 = Dense(3, activation='relu')(dense12)
output11 = Dense(1)(dense13)

input21 = Input(shape=(3, ))
dense21 = Dense(10, activation='relu')(input21)
dense22 = Dense(10, activation='relu')(dense21)
dense23 = Dense(10, activation='relu')(dense22)
output21 = Dense(1)(dense23)

merge1 = concatenate([output11, output21])
merge2 = Dense(10, name='hidden_altered1')(merge1)
merge3 = Dense(5, activation='relu', name='altered2')(merge2)
last_output = Dense(1)(merge3)

model1 = Model(inputs=[input11, input21], outputs=last_output)

# at the point where f_path is initialized, epoch, val_loss is not initialized,
# therefore epochs and val_loss can't be in f-str.
# but where f_path is called, epochs and val_loss' values are accessable.
import datetime
date = datetime.datetime.now()
date_time = date.strftime('%m%d-%H%M')
f_path = f'./_save/modelcheckpoint/keras47_{date_time}' + '_{epoch:04d}_{val_loss:.4f}.hdf5'

es = EarlyStopping(monitor='val_loss', patience=18, mode='auto', verbose=2, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=2, save_best_only=True, filepath=f_path)
model1.compile(loss='mse', optimizer='adam')
model1.fit([x1_train, x2_train], y1, batch_size=8, epochs=160, verbose=0, validation_split=0.15, callbacks=[es, mcp])
# model1.save('./_save/keras49_save_model.hd')

# Prediction n Evaluation

loss = model1.evaluate([x1_test, x2_test], y1_test)
prediction  = model1.predict([x1_test, x2_test])
r2 = r2_score(y1_test, prediction)
print('loss:', loss)
print('R2 score:', r2)