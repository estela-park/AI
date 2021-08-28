import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# 1. Data-prep
dataset = load_iris()
dataset.data = pd.DataFrame(dataset.data)
# dataset.data.keys(): RangeIndex(start=0, stop=4, step=1)
dataset.data = dataset.data.drop(columns=[0, 1])
x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, train_size=0.8, random_state=76)


# 2. Modelling
model = XGBClassifier()


# 3. Training
model.fit(x_train, y_train)


# 4. Evaluation
score = model.score(x_test, y_test)
print('model.score:', score)
print(model.feature_importances_)


'''
            <DecisionTree>
+++  max_depth=3, before deletion
model.score: 0.9
[0.         0.         0.45727133 0.54272867]
+++ after deletion
model.score: 0.9333333333333333
[0.47458126 0.52541874]

            <RandomForest>
+++  max_depth=3, before deletion
model.score: 0.9333333333333333
[0.10199735 0.02134229 0.5117467  0.36491367]
+++ after deletion
model.score: 0.9
[0.17771438 0.42779264 0.39449298]

            <GradientBoosting>
+++ Default, before deletion
model.score: 0.9666666666666667
[0.0013348  0.01934923 0.66313564 0.31618033]
+++ after deletion
model.score: 0.9666666666666667
[0.01956546 0.61729215 0.36314239]

                <XGB>
+++ Default, before deletion
model.score: 0.9333333333333333
[0.02276514 0.02512368 0.76190066 0.19021052]
+++ after deletion
model.score: 0.9666666666666667
[0.82238823 0.17761181]
'''