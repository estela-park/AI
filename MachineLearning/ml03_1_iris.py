import numpy as np
import time
from sklearn.datasets import load_iris
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

dataset = load_iris()

x = dataset.data   
# (150, 4)
y = dataset.target 
# (150, )

# in SK's model, classification task doesn't require one-hot vector for its label
# y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85)

max_abs_scaler = MaxAbsScaler()
max_abs_scaler.fit(x_train)
x_train_ma = max_abs_scaler.transform(x_train)
x_test_ma = max_abs_scaler.transform(x_test)

start = time.time()

model_ma = LinearSVC()

# Linear SVC doesn't require compiling
# model_ma.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# except data, most are set default
model_ma.fit(x_train_ma, y_train)

# evaluate => score, it gives back accuracy.
result_ma = model_ma.score(x_test_ma, y_test)
predict_ma = model_ma.predict(x_test_ma)

end = time.time() - start

# Accuracy computing
acc = accuracy_score(y_test, predict_ma)

print('it took',end)
print('accuracy score:', acc, end='')
print(', prediction for ',y_test[:8],'is',predict_ma[:8])

'''
LinearSVC
  accuracy score: 0.7391304347826086, prediction for  [2 1 0 2 0 1 1 2] is [2 1 0 1 0 2 1 2]
  accuracy score: 0.9565217391304348, prediction for  [0 1 1 0 2 0 2 2] is [0 1 2 0 2 0 2 2]
  accuracy score: 0.782608695652174, prediction for  [0 2 0 2 0 1 1 0] is [0 2 0 1 0 1 2 0]
SVC
  accuracy score: 0.9130434782608695, prediction for  [2 1 2 2 1 2 0 2] is [2 1 2 2 1 2 0 2]
  accuracy score: 0.9565217391304348, prediction for  [0 0 2 2 0 1 0 1] is [0 0 2 2 0 1 0 1]
  accuracy score: 1.0, prediction for  [1 1 1 0 1 1 2 1] is [1 1 1 0 1 1 2 1]
KNeighborsClassifier
  accuracy score: 1.0, prediction for  [1 1 0 1 2 0 2 1] is [1 1 0 1 2 0 2 1]
  accuracy score: 0.9565217391304348, prediction for  [0 0 0 1 1 0 1 1] is [0 0 0 1 1 0 1 1]
  accuracy score: 0.8695652173913043, prediction for  [0 0 0 0 0 0 1 1] is [0 0 0 0 0 0 2 1]
LogisticRegression
  accuracy score: 0.9130434782608695, prediction for  [1 2 0 2 1 0 0 1] is [1 2 0 2 1 0 0 1]
  accuracy score: 0.8695652173913043, prediction for  [2 0 1 2 2 1 2 2] is [2 0 1 2 2 1 1 2]
  accuracy score: 0.9565217391304348, prediction for  [1 2 2 0 1 1 0 0] is [1 2 2 0 1 1 0 0]
DecisionTreeClassifier
  accuracy score: 0.8695652173913043, prediction for  [0 2 2 2 2 2 2 1] is [0 2 2 2 2 2 2 2]
  accuracy score: 1.0, prediction for  [2 0 1 2 0 0 1 2] is [2 0 1 2 0 0 1 2]
  accuracy score: 0.9130434782608695, prediction for  [1 1 0 0 2 0 1 2] is [1 1 0 0 2 0 1 1]
RandomForestClassifier
  accuracy score: 1.0, prediction for  [0 2 2 0 2 1 2 2] is [0 2 2 0 2 1 2 2]
  accuracy score: 0.9130434782608695, prediction for  [1 1 1 2 2 0 0 1] is [1 1 1 1 2 0 0 1]
  accuracy score: 0.8260869565217391, prediction for  [2 2 2 0 0 2 2 1] is [1 2 2 0 0 2 2 2]
'''