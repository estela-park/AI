import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# 1. Data-prep
dataset = load_boston()
dataset.data = pd.DataFrame(dataset.data)
# dataset.data.keys(): RangeIndex(start=0, stop=13, step=1)
dataset.data = dataset.data.drop(columns=[1,2,3,6,11])
x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, train_size=0.8, random_state=76)


# 2. Modelling
model = XGBRegressor()


# 3. Training
model.fit(x_train, y_train)


# 4. Evaluation
score = model.score(x_test, y_test)
print('model.score:', score)
print(model.feature_importances_)

'''
            <DecisionTree>
+++ Default, before deletion
model.score: 0.7086555255253861
[2.21839710e-02 2.63560355e-03 4.96684880e-03 6.52459838e-06
 3.43611112e-03 2.51590824e-01 9.88350736e-03 9.30090937e-02
 4.53363548e-04 2.48546832e-02 1.33471511e-02 1.78357797e-02
 5.55796538e-01]
+++ after deletion
model.score: 0.6569603890698865
[0.02315773 0.00450827 0.26731879 0.01012862 0.09614692 0.00981387
 0.01319562 0.01615117 0.55957902]

            <RandomForest>
+++ Default, before deletion
model.score: 0.8190126080237606
[4.16025335e-02 1.34992386e-03 6.19057293e-03 4.51342150e-04
 1.03744075e-02 3.09609426e-01 1.46239053e-02 6.50985003e-02
 2.96999606e-03 1.47666126e-02 1.86083549e-02 1.26581407e-02
 5.01696285e-01]
+++ after deletion
model.score: 0.8233855441970488
[0.04092628 0.01611524 0.33032404 0.01372753 0.06435416 0.01764143
 0.01783082 0.0140266  0.48505389]

            <GradientBoosting>
+++ Default, before deletion
model.score: 0.8233435843218981
[2.99793450e-02 1.45082214e-04 1.84410827e-03 8.37543423e-04
 2.31010953e-02 3.17362513e-01 8.71514082e-03 8.63747137e-02
 1.36447080e-03 1.22732633e-02 2.75307143e-02 1.30692545e-02
 4.77402756e-01]
+++ after deletion
model.score: 0.8255373301498861
[0.02862123 0.02345135 0.31809944 0.00843108 0.08740532 0.01563589
 0.03134695 0.00998326 0.47702547]

                <XGB>
+++ Default, before deletion
model.score: 0.8058207214842088
[0.01151183 0.00153714 0.00841091 0.00100287 0.02159215 0.13647312
 0.00709863 0.05853601 0.00963087 0.03075092 0.06395119 0.00806168
 0.64144266]
+++ after deletion
model.score: 0.808992771766981
[0.02162873 0.02783262 0.13362399 0.05426992 0.01164026 0.03609728
 0.05323216 0.6616751 ]
'''