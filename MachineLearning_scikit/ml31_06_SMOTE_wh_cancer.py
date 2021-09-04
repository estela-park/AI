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

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=82)
balancer = SMOTE(random_state=96)
x_smote, y_smote = balancer.fit_resample(x_train, y_train)

print(pd.Series(y).value_counts(), pd.Series(y_train).value_counts(), pd.Series(y_smote).value_counts(), pd.Series(y_test).value_counts())
#    imbalanced label: 0    212, 1    357
#       training data: 0    159, 1    267
# training with smote: 0    267, 1    267
#           test data: 0     53, 1     90

model = XGBClassifier(n_jobs=-1)
model.fit(x_train, y_train, eval_metric='mlogloss')
y_predV = model.predict(x_test)
score_vanilla = f1_score(y_test, y_predV)

modelS = XGBClassifier(n_jobs=-1)
modelS.fit(x_smote, y_smote, eval_metric='mlogloss')
y_predS = modelS.predict(x_test)
score_smote = f1_score(y_test, y_predS)

print('imbalanced data:', score_vanilla, ', SMOTE:', score_smote)
# model.score
#   > imbalanced data: 0.958 , SMOTE: 0.972
# macroF1score
#   > imbalanced data: 0.967 , SMOTE: 0.978