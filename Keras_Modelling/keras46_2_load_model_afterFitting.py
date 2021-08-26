import time
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


# 1. data set-up
(x_train, y_train), (x_test, y_test) = mnist.load_data() 
# (50000, 28, 28, 1) (10000, 28, 28, 1) (50000, 1) (10000, 1)

x_train = x_train.reshape(60000, 28 * 28 * 1)
x_test = x_test.reshape(10000, 28 * 28 * 1)

min_max_scaler = MinMaxScaler()
x_train_mm = min_max_scaler.fit_transform(x_train).reshape(60000, 28, 28, 1)
x_test_mm = min_max_scaler.transform(x_test).reshape(10000, 28, 28, 1)

enc = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
y_train = enc.fit_transform(y_train).toarray()
y_test = enc.transform(y_test).toarray() 


# 2. modelling + compilation & training
model_mm = load_model('../_save/_keras/keras46_1.h5')


# 4. prediction & evaluation
loss_mm = model_mm.evaluate(x_test_mm, y_test)
print('entropy :', loss_mm[0], ', accuracy :', loss_mm[1])


'''
++ Where the model is saved
entropy : 0.03958873450756073 , accuracy : 0.9919000267982483

++ Where the model is loaded
entropy : 0.03958873450756073 , accuracy : 0.9919000267982483
'''