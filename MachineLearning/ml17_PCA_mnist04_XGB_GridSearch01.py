## cunsum >= 0.95

parameters = [
    {'n_estimators': [100,200,300], 'learning_rate': [0.1,0.3,0.001,0.01], 'max_depth': [4,5,6], 'tree_method': ['gpu_hist']},
    {'n_estimators': [90,100,110], 'learning_rate': [0.1,0.001,0.01], 'max_depth': [4,5,6], 'colsample_bytree': [0.6,0.9,1], 'tree_method': ['gpu_hist']},
    {'n_estimators': [90,110], 'learning_rate': [0.1,0.001,0.5], 'max_depth': [4,5,6], 'colsample_bytree': [0.6,0.9,1], 'colsample_bylevel': [0.6,0.7,0.9], 'tree_method': ['gpu_hist']},
]


import numpy as np
import time
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# (6000, 28, 28) (10000, 28, 28)

x = np.append(x_train, x_test, axis=0)
y = np.append(y_train, y_test, axis=0)
# (70000, 28, 28)

x = x.reshape(70000, 28*28)

pca = PCA(n_components=154)
x = pca.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, random_state=76)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


start = time.time()
model = GridSearchCV(XGBClassifier(), parameters, verbose=1, scoring='accuracy')
model.fit(x_train, y_train)
end = time.time() - start


score = model.score(x_test, y_test)

print('it took', end//60, 'minutes and', end%60,'seconds')
print('model.score:', score)

'''
Vanilla CNN
it took 1 minute and 10 seconds
entropy: 0.05236193165183067 accuracy: 0.9915000200271606

Vanilla DNN: stopped early at 71
it took 1 minute
entropy: 0.10611440986394882 accuracy: 0.9824761748313904

DNN with PCA 95% explanation: stopped early at 49
it took 40 seconds
entropy: 0.17585872113704681 accuracy: 0.948190450668335

GridSearch with PCA 95% explanation
'''