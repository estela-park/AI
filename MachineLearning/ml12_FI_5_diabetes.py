import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

# 1. Data-prep
dataset = load_diabetes()
dataset.data = pd.DataFrame(dataset.data)
# dataset.data.keys(): RangeIndex(start=0, stop=10, step=1)
dataset.data = dataset.data.drop(columns=[4,6,7])
x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, train_size=0.8, random_state=76)


# 2. Modelling
model = GradientBoostingRegressor()


# 3. Training
model.fit(x_train, y_train)


# 4. Evaluation
score = model.score(x_test, y_test)
print('model.score:', score)
print(model.feature_importances_)

'''
            <DecisionTree>
+++ Default, before deletion
model.score: -0.2590310742284663
[0.09550921 0.02245543 0.37755524 0.04906799 0.0434182  0.05933118
 0.05508126 0.00391002 0.22489697 0.06877451]
+++ after deletion
model.score: -0.2677022214865332
[0.10420398 0.40822272 0.07390706 0.08475534 0.25218307 0.07672782]

            <RandomForest>
+++ Default, before deletion
model.score: 0.33496957024487384
[0.06265306 0.01258196 0.31799516 0.11328125 0.04032583 0.04277201
 0.05542132 0.02247147 0.2627883  0.06970964]
+++ after deletion
model.score: 0.28014366494119736
[0.08201592 0.35588116 0.12393801 0.07856106 0.27370751 0.08589634]

            <GradientBoosting>
+++ Default, before deletion
model.score: 0.31352153303012
[0.05480722 0.01607965 0.39354152 0.09857648 0.01820125 0.05545939
 0.04452651 0.01332747 0.24649261 0.05898789]
+++ after deletion
model.score: 0.27233518103107424
[0.05747464 0.01260941 0.41322532 0.10430207 0.07199289 0.2651549
 0.07524077]
 
                <XGB>
+++ Default, before deletion
model.score: 0.12096011767011361
[0.02858655 0.06866363 0.28406927 0.06080205 0.03542177 0.03987963
 0.05920828 0.11688279 0.24256353 0.06392257]
+++ after deletion
model.score: 0.1434409983312539
[0.04072651 0.3063063  0.09584943 0.17996207 0.24839555 0.12876017]
'''
