import time
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GRU, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Data setting
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12],[20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_test = np.array([[[50],[60],[70]]])

x = x.reshape(13, 3, 1)

# Modeling
inputL = Input(shape=(3, 1))
hl = GRU(units=32, activation='relu')(inputL)
hl = Dropout(0.2)(hl)
hl = Dense(128, activation='relu')(hl)
hl = Dense(64, activation='relu')(hl)
hl = Dropout(0.2)(hl)
hl = Dense(32, activation='relu')(hl)
hl = Dense(32, activation='relu')(hl)
hl = Dense(8, activation='relu')(hl)
outputL = Dense(1)(hl)

model = Model(inputs=[inputL], outputs=[outputL])

model.summary()

# Compilation & Training
start = time.time()
es = EarlyStopping(monitor='val_loss', mode='min', patience=24, verbose=2)
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000, batch_size=3, validation_split = 0.15, callbacks=[es])
end = time.time() - start

# Prediction
result = model.predict(x_test)
print(f'prediction for [50 60 70]: {result}')
print(end)

'''    
prediction for [50 60 70]: [[81.10648]] *stopped at 88, took 7.8 seconds to fit
'''