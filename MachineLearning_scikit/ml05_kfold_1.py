import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier


# 1: Data-prep
dataset = load_iris()

x = dataset.data   
# (150, 4)
y = dataset.target 
# (150, )

# K-fold does the job
# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85)

# opt.1 Split dataset into k consecutive folds (without shuffling by default).
kfold = KFold(n_splits=5)

# opt.2 Data will be first shuffled then chopped.
# random_state controls the randomness of each fold.
# Though it provides reproducible output across multiple function calls, 
# shuffle won't happen during k times of splitting.
# kfold = KFold(n_splits=5, shuffle=True, random_state=#)


# 2: Modelling
model = RandomForestClassifier()


# 3: Training
#            > not needed


# 4: Evaluation
scores = cross_val_score(model, x, y, cv=kfold)
print('model: RandomForestClassifier, k-value: 5, accuracy:', scores, round(np.mean(scores), 4))

'''
model: LinearSVC, k-value: 5, accuracy:              [1. 1.  0.6 0.96666667 0.6] 0.86
model: SVC, k-value: 5, accuracy:                    [1. 1.  0.8 0.93333333 0.7] 0.8933
model: KNeighborsClassifier, k-value: 5, accuracy:   [1. 1.  0.8 0.93333333 0.8] 0.9133
model: LogisticRegression, k-value: 5, accuracy:     [1. 1.  0.8 0.93333333 0.8] 0.9267
model: DecisionTreeClassifier, k-value: 5, accuracy: [1. 0.9 0.9 0.93333333 0.7] 0.9067
model: RandomForestClassifier, k-value: 5, accuracy: [1. 1.  0.8 0.93333333 0.7] 0.9067
'''