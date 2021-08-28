import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

def plot_feature_importance_dataset(model):
    n_features = dataset.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), dataset.feature_names)
    plt.xlabel('Feature Importances')
    plt.ylabel('Features')
    plt.ylim(-1, n_features)


# 1. Data-prep
dataset = load_boston()
x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, train_size=0.8, random_state=76)


# 2. Modelling
model = GradientBoostingRegressor()


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
model.score: 0.7086555255253861
[2.21839710e-02 2.63560355e-03 4.96684880e-03 6.52459838e-06
 3.43611112e-03 2.51590824e-01 9.88350736e-03 9.30090937e-02
 4.53363548e-04 2.48546832e-02 1.33471511e-02 1.78357797e-02
 5.55796538e-01]
+++  max_depth=4
model.score: 0.6342359588868556
[0.01298706 0.         0.         0.         0.         0.27589788
 0.         0.09288892 0.         0.         0.00480264 0.01066924
 0.60275426]
+++  max_depth=3
model.score: 0.4864857419643257
[0.         0.         0.         0.         0.         0.23060822
 0.         0.10118465 0.         0.         0.         0.01162209
 0.65658505]
+++  max_depth=5
model.score: 0.7085222641711371
[0.01665203 0.         0.00142349 0.         0.00224726 0.27369287
 0.0014449  0.09152579 0.         0.00243604 0.01218223 0.01158925
 0.58680614]
            <RandomForest>
+++  WITHOUT args
model.score: 0.8190126080237606
[4.16025335e-02 1.34992386e-03 6.19057293e-03 4.51342150e-04
 1.03744075e-02 3.09609426e-01 1.46239053e-02 6.50985003e-02
 2.96999606e-03 1.47666126e-02 1.86083549e-02 1.26581407e-02
 5.01696285e-01]
+++  max_depth=3
model.score: 0.7155514406977036
[3.34430250e-02 7.17037580e-04 8.55049210e-04 3.35010921e-04
 5.73777762e-03 3.33863006e-01 1.61218970e-03 5.61923692e-02
 3.36094824e-03 4.10447678e-03 6.50686790e-03 3.53626134e-03
 5.49735980e-01]
+++  max_depth=4
model.score: 0.7726042096606984
[3.88549013e-02 2.93675840e-05 2.63004254e-03 1.46696699e-04
 8.47484979e-03 2.95761643e-01 6.59746284e-03 5.48877525e-02
 3.27177907e-03 7.33894371e-03 1.02683924e-02 5.60133659e-03
 5.66136832e-01]
+++  max_depth=5
model.score: 0.779685166602704
[3.98458962e-02 8.94586113e-04 2.73958259e-03 1.95109625e-04
 9.89756556e-03 3.42255984e-01 8.14933179e-03 5.97220041e-02
 3.89487233e-03 1.19685817e-02 1.24657964e-02 9.67552221e-03
 4.98295167e-01]
                <XGB>
model.score: 0.8058207214842088
[0.01151183 0.00153714 0.00841091 0.00100287 0.02159215 0.13647312
 0.00709863 0.05853601 0.00963087 0.03075092 0.06395119 0.00806168
 0.64144266]
            <GradientBoosting>
model.score: 0.8233435843218981
[2.99793450e-02 1.45082214e-04 1.84410827e-03 8.37543423e-04
 2.31010953e-02 3.17362513e-01 8.71514082e-03 8.63747137e-02
 1.36447080e-03 1.22732633e-02 2.75307143e-02 1.30692545e-02
 4.77402756e-01]
'''