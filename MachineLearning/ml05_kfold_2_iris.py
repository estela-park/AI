import numpy as np
import time
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# 1: Data-prep
dataset = load_iris()

x = dataset.data   
# (150, 4)
y = dataset.target 
# (150, )

# These can be implemented together.
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, random_state=76)
kfold = KFold(n_splits=5, shuffle=True, random_state=99)


# 2: Modelling
model = SVC()


# 3: Training
#            > not needed


# 4: Evaluation
scores = cross_val_score(model, x_train, y_train, cv=kfold)
# y_pre = cross_val_predict(model, x_test)
# acc = accuracy_score(y_test, y_pre)
print('model: SVC, k-value: 5, accuracy:', scores, round(np.mean(scores), 4))
# print('test score:',acc)

'''
model: LinearSVC,               k-value: 5, accuracy: [0.96153846 1.         0.92       0.92       0.92      ] 0.9443
model: SVC,                     k-value: 5, accuracy: [0.96153846 1.         1.         0.92       1.        ] 0.9763
model: KNeighborsClassifier,    k-value: 5, accuracy: [1.   1.   1.   0.92 0.96]                               0.976
model: LogisticRegression,      k-value: 5, accuracy: [0.96153846 1.         0.96       0.92       1.        ] 0.9683
model: DecisionTreeClassifier,  k-value: 5, accuracy: [0.92307692 0.96153846 0.96       0.88       1.        ] 0.9449
model: RandomForestClassifier,  k-value: 5, accuracy: [0.92307692 0.96153846 0.96       0.88       0.96      ] 0.9369
'''