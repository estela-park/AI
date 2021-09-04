import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')


dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

y = np.array(y).reshape(569, 1)
data = np.concatenate((x, y), axis=1)
x = data[112:, :-1]
y = data[112:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=82)
balancer = SMOTE(random_state=96)
x_smote, y_smote = balancer.fit_resample(x_train, y_train)

print(pd.Series(y).value_counts(), pd.Series(y_train).value_counts(), pd.Series(y_smote).value_counts(), pd.Series(y_test).value_counts())
#    imbalanced label: 0    144, 1    313
#       training data: 0    108, 1    234
# training with smote: 0    234, 1    234
#           test data: 0     36, 1     79

model = XGBClassifier(n_jobs=-1)
model.fit(x_train, y_train, eval_metric='mlogloss')
score_vanilla = model.score(x_test, y_test)
y_predV = model.predict(x_test)
f1_vanilla = f1_score(y_test, y_predV)

modelS = XGBClassifier(n_jobs=-1)
modelS.fit(x_smote, y_smote, eval_metric='mlogloss')
score_smote = modelS.score(x_test, y_test)
y_predS = modelS.predict(x_test)
f1_smote = f1_score(y_test, y_predS)

print('imbalanced data:', score_vanilla, ', SMOTE:', score_smote)
print('imbalanced data:', f1_vanilla, ', SMOTE:', f1_smote)
# model.score
#   > imbalanced data: 0.9826 , SMOTE: 0.9652
# macroF1score
#   > imbalanced data: 0.9875 , SMOTE: 0.9743