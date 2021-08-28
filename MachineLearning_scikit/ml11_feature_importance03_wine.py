import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

def plot_feature_importance_dataset(model):
    n_features = dataset.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), dataset.feature_names)
    plt.xlabel('Feature Importances')
    plt.ylabel('Features')
    plt.ylim(-1, n_features)


# 1. Data-prep
dataset = load_wine()
x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, train_size=0.8, random_state=76)


# 2. Modelling
model = XGBClassifier()


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
model.score: 0.9444444444444444
[0.         0.         0.02089181 0.         0.         0.
 0.40351866 0.         0.         0.4345252  0.         0.
 0.14106433]
+++  max_depth=3
model.score: 0.9444444444444444
[0.         0.         0.02089181 0.         0.         0.
 0.40351866 0.         0.         0.4345252  0.         0.
 0.14106433]
+++  max_depth=4
model.score: 0.9444444444444444
[0.03214124 0.         0.         0.         0.         0.
 0.40351866 0.         0.         0.40238396 0.02089181 0.
 0.14106433]
+++  max_depth=5
model.score: 0.9444444444444444
[0.03214124 0.         0.02089181 0.         0.         0.
 0.40351866 0.         0.         0.40238396 0.         0.
 0.14106433]
            <RandomForest>
+++  WITHOUT args
model.score: 0.9722222222222222
[0.12672803 0.03716039 0.01624732 0.0200954  0.020553   0.05257941
 0.14621849 0.0171439  0.0184531  0.15520619 0.097395   0.12362811
 0.16859165]
+++  max_depth=3
model.score: 0.9722222222222222
[0.13475014 0.02464608 0.01453546 0.01416367 0.01688242 0.04011164
 0.17440027 0.0158278  0.01606794 0.18059685 0.07494148 0.13302117
 0.16005508]
+++  max_depth=4
model.score: 0.9722222222222222
[0.13073734 0.03592788 0.01633676 0.02090166 0.02278045 0.06253318
 0.16073205 0.00965039 0.01890901 0.16159569 0.07785832 0.12749461
 0.15454267]
+++  max_depth=5
model.score: 0.9722222222222222
[0.17524836 0.04564446 0.01262786 0.01647631 0.02721297 0.0487403
 0.17551221 0.01809206 0.01920316 0.16016641 0.08068926 0.09773715
 0.12264948]
            <GradientBoosting>
model.score: 0.9722222222222222
[1.40702595e-02 4.87337532e-02 4.54933237e-03 1.11734672e-03
 5.01247371e-03 2.62972980e-04 2.95588979e-01 3.28482591e-06
 2.68713717e-03 3.17453329e-01 1.60571066e-02 2.88339182e-02
 2.65630107e-01]
                <XGB>
model.score: 0.9722222222222222
[0.03388827 0.05916279 0.01936584 0.00547369 0.01248416 0.0127115
 0.23918746 0.         0.01338424 0.2579099  0.05107394 0.0854165
 0.2099417 ]
'''