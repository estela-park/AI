from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.models import load_model
import time

datasets = load_diabetes()

x = datasets.data   # (442, 10)
y = datasets.target # (442, )

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85)

model = load_model('./_save/keras46_1_save_mode_1.h5')

start = time.time()
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=800, validation_split=0.15, verbose=2, batch_size=32)

loss = model.evaluate(x_test, y_test)
predict = model.predict(x_test)

end = time.time() - start

model.save('./_save/keras46_2_save_model_1.h5')

r2 = r2_score(y_test, predict)

print('loss:', loss, 'actual data:', y_test, 'machine predicted:', predict)
print('accuracy:', r2)

'''
loss: [mse: 3567.356689453125, mae: 49.89421081542969] accuracy: 0.3960380554636723
'''

# pre-compilation n training model doesn't have set weitght in it
# post-compilation n training model has set weitght in it
model1 = load_model('./_save/keras46_1_save_model_1.h5')
model2 = load_model('./_save/keras46_2_save_model_1.h5')
