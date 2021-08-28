import time
from sklearn.datasets import load_wine
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

dataset = load_wine()

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

model_ma = LinearSVC()

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
    accuracy score: 0.9629629629629629, prediction for  [0 0 1 1 2 2 0 2] is [0 0 1 1 2 2 0 2]
**SVC
    accuracy score: 0.9259259259259259, prediction for  [0 0 2 1 1 0 0 1] is [0 0 2 1 1 0 0 1]
**KNeighborsClassifier
    accuracy score: 1.0, prediction for  [1 1 0 1 1 1 2 2] is [1 1 0 1 1 1 2 2]
**LogisticRegression
    accuracy score: 1.0, prediction for  [0 2 0 0 2 0 1 1] is [0 2 0 0 2 0 1 1]
**DecisionTreeClassifier
    accuracy score: 0.9259259259259259, prediction for  [0 0 1 0 0 0 0 2] is [0 0 1 0 0 0 0 2]
**RandomForestClassifier
    accuracy score: 1.0, prediction for  [2 2 0 0 2 0 2 2] is [2 2 0 0 2 0 2 2]
'''