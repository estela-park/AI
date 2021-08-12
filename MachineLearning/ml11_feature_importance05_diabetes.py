import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

def plot_feature_importance_dataset(model):
    n_features = dataset.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), dataset.feature_names)
    plt.xlabel('Feature Importances')
    plt.ylabel('Features')
    plt.ylim(-1, n_features)


# 1. Data-prep
dataset = load_diabetes()
x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, train_size=0.8, random_state=76)


# 2. Modelling
model = DecisionTreeRegressor()


# 3. Training
model.fit(x_train, y_train)


# 4. Evaluation
score = model.score(x_test, y_test)
print('model.score:', score)
print(model.feature_importances_)

plot_feature_importance_dataset(model)
plt.show()


'''
            <DecisionTree>
+++  WITHOUT args
model.score: -0.2590310742284663
[0.09550921 0.02245543 0.37755524 0.04906799 0.0434182  0.05933118
 0.05508126 0.00391002 0.22489697 0.06877451]
+++  max_depth=4
model.score: 0.2139826017800449
[0.         0.         0.64688752 0.02173257 0.         0.03626624
 0.02537793 0.         0.26973574 0.        ]
+++  max_depth=3
model.score: 0.248529478678701
[0.02103787 0.01580793 0.56205071 0.01888243 0.         0.03151006
 0.03762198 0.         0.27497255 0.03811648]
+++  max_depth=5
model.score: 0.0795226834213637
[0.07402842 0.01363722 0.49614986 0.02963949 0.01681202 0.02753847
 0.03288004 0.         0.26243937 0.04687511]
            <RandomForest>
+++  WITHOUT args
model.score: 0.33496957024487384
[0.06265306 0.01258196 0.31799516 0.11328125 0.04032583 0.04277201
 0.05542132 0.02247147 0.2627883  0.06970964]
+++  max_depth=3
model.score: 0.3844098311550149
[1.65036394e-02 3.70572974e-04 4.01592353e-01 8.67612980e-02
 6.04244844e-03 1.76798535e-02 1.49092926e-02 1.06687517e-02
 4.07479683e-01 3.79921072e-02]
+++  max_depth=4
model.score: 0.37021299701683763
[0.0338978  0.00340924 0.38013859 0.09018813 0.01471332 0.02693983
 0.02627985 0.02330583 0.35654343 0.04458396]
+++  max_depth=5
model.score: 0.36555850344993657
[0.04329421 0.00745429 0.3847031  0.10142439 0.02483789 0.03372803
 0.03105281 0.0274047  0.29271125 0.05338934]
                <XGB>
model.score: 0.12096011767011361
[0.02858655 0.06866363 0.28406927 0.06080205 0.03542177 0.03987963
 0.05920828 0.11688279 0.24256353 0.06392257]
            <GradientBoosting>
model.score: 0.31352153303012
[0.05480722 0.01607965 0.39354152 0.09857648 0.01820125 0.05545939
 0.04452651 0.01332747 0.24649261 0.05898789]
'''