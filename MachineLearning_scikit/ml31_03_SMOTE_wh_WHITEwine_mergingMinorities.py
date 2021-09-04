# ****remember how list operation was used
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')


dataset = pd.read_csv('../_data/white_wine.csv', sep=';', index_col=None, header=0)
dataset = dataset.values

x = dataset[:, :11]
y = dataset[:, 11]
# (4898, 11) (4898,)
# 3.0      20
# 4.0     163
# 5.0    1457
# 6.0    2198
# 7.0     880
# 8.0     175
# 9.0       5

newdata = []
for i in list(y):
    if i <= 5:
        newdata += [0]
    elif i == 6:
        newdata += [1]
    else: 
        newdata += [2]
x_train, x_test, y_train, y_test = train_test_split(x, newdata, stratify=newdata, random_state=32)
print(pd.Series(y_train).value_counts(), pd.Series(y_test).value_counts())

# 0    1230     0    410
# 1    1648     1    550
# 2     795     2    265

model = XGBClassifier(n_jobs=-1)
model.fit(x_train, y_train, eval_metric='mlogloss')
score_vanilla = model.score(x_test, y_test)

balancer = SMOTE(random_state=96, k_neighbors=100)
x_smote, y_smote = balancer.fit_resample(x_train, y_train)

modelS = XGBClassifier(n_jobs=-1)
modelS.fit(x_smote, y_smote, eval_metric='mlogloss')
score_smote = modelS.score(x_test, y_test)

print('imbalanced data:', score_vanilla, ', SMOTE:', score_smote)
# imbalanced data: 0.66, minorities merged: 0.70 , SMOTE(k_neighbors=5): 0.69, SMOTE(k_neighbors=100): 0.686530612244898