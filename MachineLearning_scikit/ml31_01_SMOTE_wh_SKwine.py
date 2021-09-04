# [Reference: https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification]
# 
# SMOTE(sampling_strategy='auto', random_state=None, k_neighbors=5, n_jobs=None)
#     >>performs over-sampling using SMOTE-kMeans: Default
#     >>nominalVScontinuous
#     >>borderlineVSk-neighbors
#     **sampling_strategy: float, str, dict or callable, default=’auto’
#                          how to resample the data set.
#       > float:
#         >> the desired ratio of the number of samples in the minority class after resampling / the number of samples in the majority class.
#         >> only available for binary classification. An error is raised for multi-class classification.
#       > str: The number of samples in the different classes will be equalized.
#         >> how to achieve that
#             > 'minority': resample only the minority class;
#             > 'not minority': resample all classes but the minority class;
#             > 'not majority': resample all classes but the majority class;
#             > 'all': resample all classes;
#             > 'auto': equivalent to 'not majority'.
#       > dict:
#           >> the keys correspond to the targeted classes.
#           >> The values correspond to the desired number of samples for each targeted class.
#     **k_neighbors: int or object, default=5
#       > int: number of nearest neighbours to used to construct synthetic samples.
#     **n_jobs: int, default=None
#       > Number of CPU cores used during the cross-validation loop.
#           >> None means 1
#           >> -1 means using all processors
# 
# SMOTENC():         Synthetic Minority Over-sampling Technique for Nominal and Continuous.
# SMOTEN():          Synthetic Minority Over-sampling Technique for Nominal.
# BorderlineSMOTE(): Over-sampling using Borderline SMOTE.
# KMeansSMOTE():     Apply a KMeans clustering before to over-sample using SMOTE.


import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')


dataset = load_wine()
x = dataset.data
y = dataset.target
# (178, 13) (178, )
print(pd.Series(y).value_counts())
# 0    59
# 1    71
# 2    48

# data is aligned numerically
x_cropped = x[:-30]
y_cropped = y[:-30]
print(pd.Series(y_cropped).value_counts())
# 0    59
# 1    71
# 2    18

# stratify argument ensures that both the train and test sets have the same proportion of examples in each class
# [Without stratify]
#  > original: {0: 94, 1: 6}, train: {0: 45, 1: 5}, test:: {0: 49, 1: 1}
# [With stratify]
#  > original: {0: 94, 1: 6}, train: {0: 47, 1: 3}, test:: {0: 47, 1: 3}
x_train, x_test, y_train, y_test = train_test_split(x_cropped, y_cropped, stratify=y_cropped, random_state=82)
print(pd.Series(y_train).value_counts(), pd.Series(y_test).value_counts())
# [Without stratify]
# train: 0    47    test: 0    12
#        1    50          1    21
#        2    14          2     4
# [With stratify]
# train:  0    44   test: 0    15
#         1    53         1    18
#         2    14         2     4

model = XGBClassifier(n_jobs=-1)
model.fit(x_train, y_train, eval_metric='mlogloss')
score_vanilla = model.score(x_test, y_test)

balancer = SMOTE(random_state=96)
x_smote, y_smote = balancer.fit_resample(x_train, y_train)
# x_train.shape; (111, 13), x_smote.shape: (159, 13) -> 159 = 53*3

modelS = XGBClassifier(n_jobs=-1)
modelS.fit(x_smote, y_smote, eval_metric='mlogloss')
score_smote = modelS.score(x_test, y_test)

print('imbalanced data:', score_vanilla, ', SMOTE:', score_smote)
# imbalanced data: 0.972972972972973 , SMOTE: 0.9459459459459459