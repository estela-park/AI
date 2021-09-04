# with a different strategy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')


dataset = pd.read_csv('../_data/white_wine.csv', sep=';', index_col=None, header=0)
dataset = dataset.values

x = dataset[:, :11]
y = dataset[:, 11]
# (4898, 11) (4898,)

newdata = []
for i in list(y):
    if i==3:
        newdata += [0]
    elif i == 4:
        newdata += [0]
    elif i == 5:
        newdata += [1]
    elif i == 6:
        newdata += [2]
    elif i == 7:
        newdata += [3]
    elif i == 8:
        newdata += [0]
    elif i == 9:
        newdata += [0]
x_train, x_test, y_train, y_test = train_test_split(x, newdata, stratify=newdata, random_state=32)
print(pd.Series(y_train).value_counts(), pd.Series(y_test).value_counts())
# 0     272     0     91
# 1    1093     1    364
# 2    1648     2    550
# 3     660     3    220


model = XGBClassifier(n_jobs=-1)
model.fit(x_train, y_train, eval_metric='mlogloss')
y_predV = model.predict(x_test)
score_vanilla = f1_score(y_test, y_predV, average='macro')

balancer = SMOTE(random_state=96, k_neighbors=100)
x_smote, y_smote = balancer.fit_resample(x_train, y_train)

modelS = XGBClassifier(n_jobs=-1)
modelS.fit(x_smote, y_smote, eval_metric='mlogloss')
y_predS = modelS.predict(x_test)
score_smote = f1_score(y_test, y_predS, average='macro')

print('default:', score_vanilla, ', SMOTE:', score_smote)
# macroF1score
# default: 0.608059606709587 , SMOTE: 0.6014386864525249