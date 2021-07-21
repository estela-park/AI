from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MaxAbsScaler, RobustScaler, PowerTransformer, QuantileTransformer
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

dataset = load_breast_cancer()

# print(dataset.DESCR)
# print(dataset.feature_names)

x = dataset.data   # (569, 30)
y = dataset.target # (569, )/ y.unique(): [0, 1]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

'''
for robust, accuracy: [0.41252321004867554, 0.9824561476707458]

'''