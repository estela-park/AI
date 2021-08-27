import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, Adamax, Adadelta, Adagrad
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam


x = np.array(range(1, 11))
y = np.array(range(1, 11))

model = Sequential()
model.add(Dense(1000, input_dim=1))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

optimizer = Adadelta(lr=0.1)


model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
model.fit(x, y, batch_size=1, epochs=20)

loss, mse = model.evaluate(x, y, batch_size=1)
y_pred = model.predict([11])

print('***** for Nadam with learning rate=0.001 *****')
print('loss:', loss, ', prediction:', y_pred)

'''
epo=100
loss: 5.127986923980643e-11 , prediction: [[10.99999]]
epo=10
loss: 0.0010659139370545745 , prediction: [[10.96064]]
epo=5
loss: 0.16454333066940308 , prediction: [[10.210579]]

tf.keras.optimizers.Adam(learning_rate=0.001, ...)
***** for Adam with learning rate=0.001 *****
loss: 0.026105787605047226 , prediction: [[10.687069]]
***** for Adam with learning rate=0.01 *****
loss: 2.7784099643213267e-07 , prediction: [[10.998879]]
***** for Adam with learning rate=0.1 *****
loss: 24663.216796875 , prediction: [[105.58288]]

tf.keras.optimizers.Adamax(learning_rate=0.001, ...)
***** for Adamax with learning rate=0.001 *****
loss: 0.00017715987632982433 , prediction: [[10.978465]]
***** for Adamax with learning rate=0.01 *****
loss: 0.0015830111224204302 , prediction: [[11.037195]]
***** for Adamax with learning rate=0.1 *****
loss: 189.91629028320312 , prediction: [[31.817015]]

tf.keras.optimizers.Adadelta(learning_rate=0.001, ...)
***** for Adadelta with learning rate=0.001 *****
loss: 26.977741241455078 , prediction: [[1.7821175]]
***** for Adadelta with learning rate=0.01 *****
loss: 1.1931507587432861 , prediction: [[8.975206]]
***** for Adadelta with learning rate=0.1 *****
loss: 0.03982121869921684 , prediction: [[11.344234]]

tf.keras.optimizers.Adagrad(learning_rate=0.001, ...)
***** for Adagrad with learning rate=0.001 *****
loss: 0.001547129126265645 , prediction: [[10.951429]]
***** for Adagrad with learning rate=0.01 *****
loss: 0.00030073203379288316 , prediction: [[11.026133]]
***** for Adagrad with learning rate=0.1 *****
loss: 2907.410888671875 , prediction: [[-60.2117]]

tf.keras.optimizers.RMSprop(learning_rate=0.001, ...)
***** for RMSprop with learning rate=0.001 *****
loss: 0.04970758408308029 , prediction: [[10.5597925]]
***** for RMSprop with learning rate=0.01 *****
loss: 2.0885279178619385 , prediction: [[7.878186]]
***** for RMSprop with learning rate=0.1 *****
loss: 5440145.5 , prediction: [[-5027.5376]]

tf.keras.optimizers.SGD(learning_rate=0.01, ...)
***** for SGD with learning rate=0.001 *****
loss: 0.002552452264353633 , prediction: [[10.892016]]
***** for SGD with learning rate=0.01 *****
loss: nan , prediction: [[nan]]
***** for SGD with learning rate=0.1 *****
loss: nan , prediction: [[nan]]

tf.keras.optimizers.Nadam(learning_rate=0.001, ...)
***** for Nadam with learning rate=0.001 *****
loss: 0.012259592302143574 , prediction: [[10.829342]]
***** for Nadam with learning rate=0.01 *****
loss: 0.6542525291442871 , prediction: [[9.699055]]
***** for Nadam with learning rate=0.1 *****
loss: 10007.357421875 , prediction: [[-148.08563]]
'''