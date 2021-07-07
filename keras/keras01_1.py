from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1. Data
x = np.array([1, 2, 3])
y = np.array([1, 2, 3])

# 2. Model
model = Sequential()
model.add(Dense(1, input_dim=1))

# 3. Compiling & Training
model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=4000, batch_size=1)

# 4. Evaluation & Prediction
loss = model.evaluate(x, y)
print('loss: '+ str(loss))

result = model.predict([4])
print('prediction for 4: '+ str(result))