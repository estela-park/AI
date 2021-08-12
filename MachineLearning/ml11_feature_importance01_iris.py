import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def plot_feature_importance_dataset(model):
    n_features = dataset.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), dataset.feature_names)
    plt.xlabel('Feature Importances')
    plt.ylabel('Features')
    plt.ylim(-1, n_features)


# 1. Data-prep
dataset = load_iris()
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
model.score: 0.9333333333333333
[0.02508361 0.         0.45608735 0.51882904]
+++  max_depth=4
model.score: 0.9
[0.         0.03412969 0.93646713 0.02940318]
+++  max_depth=3
model.score: 0.9
[0.         0.         0.45727133 0.54272867]
+++  max_depth=5
model.score: 0.9
[0.00762066 0.0338696  0.92933063 0.02917911]
            <RandomForest>
+++  WITHOUT args
model.score: 0.9666666666666667
[0.09762117 0.03454616 0.44282262 0.42501004]
+++  max_depth=3
model.score: 0.9333333333333333
[0.10199735 0.02134229 0.5117467  0.36491367]
+++  max_depth=4
model.score: 0.9666666666666667
[0.11023986 0.02809821 0.45395344 0.40770849]
+++  max_depth=5
model.score: 0.9333333333333333
[0.07868965 0.03458229 0.51451558 0.37221248]
            <GradientBoosting>
model.score: 0.9666666666666667
[0.0013348  0.01934923 0.66313564 0.31618033]
                <XGB>
model.score: 0.9333333333333333
[0.02276514 0.02512368 0.76190066 0.19021052]
'''