import numpy as np
import time
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

dataset = load_breast_cancer()

x = dataset.data  
print(x.shape) 
# (150, 4)
y = dataset.target 
print(y.shape)
# (150, )

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85)

max_abs_scaler = MaxAbsScaler()
max_abs_scaler.fit(x_train)
x_train_ma = max_abs_scaler.transform(x_train)
x_test_ma = max_abs_scaler.transform(x_test)

start = time.time()

model_ma = RandomForestClassifier()

# except data, most parameters are set default
model_ma.fit(x_train_ma, y_train)

result_ma = model_ma.score(x_test_ma, y_test)
predict_ma = model_ma.predict(x_test_ma)

end = time.time() - start

# Accuracy computing
acc = accuracy_score(y_test, predict_ma)

print('it took',end)
print('accuracy score:', acc, end='')
print(', prediction for ',y_test[:8],'is',predict_ma[:8])

'''
**LinearSVC
    accuracy score: 0.9302325581395349, prediction for  [0 0 0 1 0 0 1 1] is [0 0 0 1 0 0 1 1]
**SVC
    accuracy score: 0.9651162790697675, prediction for  [0 1 1 0 1 1 1 1] is [0 1 1 0 1 1 1 1]
**KNeighborsClassifier
    accuracy score: 0.9767441860465116, prediction for  [1 1 1 1 0 0 1 1] is [1 1 1 1 0 0 1 1]
**LogisticRegression
    accuracy score: 0.9534883720930233, prediction for  [1 0 0 1 1 1 1 0] is [1 0 0 1 1 1 1 0]
**DecisionTreeClassifier
    accuracy score: 0.9186046511627907, prediction for  [0 0 1 0 0 1 1 1] is [0 0 1 0 0 1 1 1]
**RandomForestClassifier
    accuracy score: 0.9767441860465116, prediction for  [0 1 1 1 1 1 0 1] is [0 1 1 1 1 1 0 1]
'''