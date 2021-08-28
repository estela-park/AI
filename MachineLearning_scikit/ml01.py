import numpy as np
import time
from sklearn.datasets import load_iris
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

dataset = load_iris()

x = dataset.data   
# (150, 4)
y = dataset.target 
# (150, )

# in SK's model, classification task doesn't require one-hot vector for its label
# y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85) # , random_state=72

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
it took 0.001 second, and was scaled with maxabs scaler.
  ++ below are accuracy calculated with model.score
     <classification score is set to accuracy score>
     <regression score is to R^2>
accuracy: 0.9565217391304348
accuracy: 0.9565217391304348, prediction for  [0 1 1 1 0 1 2 0] is [0 1 1 1 0 1 2 0]
accuracy: 0.9565217391304348, prediction for  [2 2 2 1 0 2 0 0] is [2 1 2 1 0 2 0 0]
  ++ below is accuracy calculated with metrics accuracy_score
accuracy score: 0.9565217391304348, prediction for  [2 2 0 1 1 1 2 2] is [2 2 0 1 1 1 2 2]
'''