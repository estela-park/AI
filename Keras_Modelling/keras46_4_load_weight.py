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
model_mm_b = load_model('../_save/_keras/keras46_3.h5')
model_mm_b.load_weights('../_save/_keras/keras46_3_beforeF.h5')

model_mm_a = load_model('../_save/_keras/keras46_3.h5')
model_mm_a.load_weights('../_save/_keras/keras46_3_afterF.h5')


# 4. prediction & evaluation
loss_mm_b = model_mm_b.evaluate(x_test_mm, y_test)
print('entropy :', loss_mm_b[0], ', accuracy :', loss_mm_b[1])
print('============================================')
loss_mm_a = model_mm_a.evaluate(x_test_mm, y_test)
print('entropy :', loss_mm_a[0], ', accuracy :', loss_mm_a[1])

'''
+++ Where the weights was saved
entropy : 0.0494924858212471 , accuracy : 0.9901999831199646
+++ before fitting
entropy : 2.2978055477142334 , accuracy : 0.1363999992609024
+++ after fitting
entropy : 0.0494924858212471 , accuracy : 0.9901999831199646
'''