import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


# 1. Data-prep
dataset = load_wine()
# type(dataset.data): <class 'numpy.ndarray'>
dataset.data = pd.DataFrame(dataset.data)
# dataset.data.keys(): RangeIndex(start=0, stop=13, step=1)
dataset.data = dataset.data.drop(columns=[2, 3, 4, 5, 7, 8])
# dataset.data = np.array(dataset.data)
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
+++ Default, before deletion
model.score: 0.9444444444444444
[0.         0.         0.02089181 0.         0.         0.
 0.40351866 0.         0.         0.4345252  0.         0.
 0.14106433]
+++ after deletion
model.score: 0.9444444444444444
[0.02089181 0.40351866 0.4345252  0.14106433]

            <RandomForest>
+++ Default, before deletion
model.score: 0.9722222222222222
[0.12672803 0.03716039 0.01624732 0.0200954  0.020553   0.05257941
 0.14621849 0.0171439  0.0184531  0.15520619 0.097395   0.12362811
 0.16859165]
 +++ after deletion
model.score: 0.9722222222222222
[0.10675567 0.02968779 0.01815691 0.03162901 0.20648374 0.22563207
 0.08048234 0.1341417  0.16703077]

            <GradientBoosting>
+++ Default, before deletion
model.score: 0.9722222222222222
[1.40702595e-02 4.87337532e-02 4.54933237e-03 1.11734672e-03
 5.01247371e-03 2.62972980e-04 2.95588979e-01 3.28482591e-06
 2.68713717e-03 3.17453329e-01 1.60571066e-02 2.88339182e-02
 2.65630107e-01]
+++ after deletion
model.score: 0.9722222222222222
[0.01103292 0.05207659 0.29618803 0.32496039 0.02083211 0.02555122
 0.26935873]

                <XGB>
+++ Default, before deletion
model.score: 0.9722222222222222
[0.03388827 0.05916279 0.01936584 0.00547369 0.01248416 0.0127115
 0.23918746 0.         0.01338424 0.2579099  0.05107394 0.0854165
 0.2099417 ]
+++ after deletion
model.score: 0.9722222222222222
[0.0281595  0.06570302 0.25346467 0.28067163 0.05066719 0.10425981
 0.2170741 ]
'''