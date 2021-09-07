import time
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


datasets = load_diabetes()

x = datasets.data
y = datasets.target 

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, random_state=78)

model = Sequential()
model.add(Dense(128 , input_shape=(10,)))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(1))

start = time.time()
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
es = EarlyStopping(monitor='val_loss', patience=17, mode='min', verbose=2)
cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, filepath='../_save/ModelCheckPoint/keras47_MCP.hdf5')
model.fit(x_train, y_train, epochs=360, batch_size=16, validation_split=0.15) # , callbacks=[es, cp])
# model.save('../_save/ModelCheckPoint/keras47_MCP_model.h5')
loss = model.evaluate(x_test, y_test)
predict = model.predict(x_test)
end = time.time() - start

r2 = r2_score(y_test, predict)

print('loss:', loss, 'actual data:', y_test, 'machine predicted:', predict, 'accuracy:', r2)

'''
_saved at random_state=78
loss: [2748.679931640625, 44.2568244934082] accuracy: 0.5462202791204794  
'''