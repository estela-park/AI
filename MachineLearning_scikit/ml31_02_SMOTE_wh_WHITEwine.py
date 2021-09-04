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
print(pd.Series(y).value_counts())
# 3.0      20
# 4.0     163
# 5.0    1457
# 6.0    2198
# 7.0     880
# 8.0     175
# 9.0       5

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=32)
print(pd.Series(y_train).value_counts(), pd.Series(y_test).value_counts())
# [Without stratify]
# 3.0      13       3.0      7
# 4.0     122       4.0     41
# 5.0    1099       5.0    358
# 6.0    1640       6.0    558
# 7.0     664       7.0    216
# 8.0     130       8.0     45
# 9.0       5
#
# [With stratify]
# 3.0      15       3.0      5
# 4.0     122       4.0     41
# 5.0    1093       5.0    364
# 6.0    1648       6.0    550
# 7.0     660       7.0    220
# 8.0     131       8.0     44
# 9.0       4       9.0      1

model = XGBClassifier(n_jobs=-1)
model.fit(x_train, y_train, eval_metric='mlogloss')
score_vanilla = model.score(x_test, y_test)

balancer = SMOTE(random_state=96, k_neighbors=3)
# Expected n_neighbors <= n_samples,  but n_samples = 4, n_neighbors = 6
# class9 has 4 data-points, it can't pick up 6 out of 4
# n_neighbors = k_neighbors + 1 -> data-point a and its k neighbors
x_smote, y_smote = balancer.fit_resample(x_train, y_train)

modelS = XGBClassifier(n_jobs=-1)
modelS.fit(x_smote, y_smote, eval_metric='mlogloss')
score_smote = modelS.score(x_test, y_test)

print('imbalanced data:', score_vanilla, ', SMOTE:', score_smote)
# imbalanced data: 0.6669387755102041 , SMOTE: 0.6514285714285715