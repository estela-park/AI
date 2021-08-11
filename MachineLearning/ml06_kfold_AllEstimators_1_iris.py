import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.utils import all_estimators


# 1. Data-prep
dataset = load_iris()

x = dataset.data   
y = dataset.target 

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85) 
kfold = KFold(n_splits=5, shuffle=True, random_state=99)


# 2, 3, 4. Modelling, Training, Evaluation
all_model = all_estimators('classifier')
count = 0
for name, model in all_model:
    try: 
        model = model()
        scores = cross_val_score(model, x_train, y_train, cv=kfold)
        print(f'with {name}, ', end='')
        print('k-value: 5, accuracy:', scores, round(np.mean(scores), 4))

        count += 1

    except:
        print(f'{name} will not work for this particular data-set')
        continue

print('the number of working models:', count)
print('the number of candidate models:', len(all_model))

'''
with AdaBoostClassifier,             k-value: 5, accuracy: [0.92307692 1.         0.96     0.92     0.92    ] 0.9446
with BaggingClassifier,              k-value: 5, accuracy: [0.92307692 1.         0.96     0.92     0.92    ] 0.9446
with BernoulliNB,                    k-value: 5, accuracy: [0.30769231 0.30769231 0.24     0.32     0.36    ] 0.3071
with CalibratedClassifierCV,         k-value: 5, accuracy: [0.84615385 0.96153846 0.92     0.92     0.88    ] 0.9055
with CategoricalNB,                  k-value: 5, accuracy: [1.         0.96153846 0.88     0.96     0.84    ] 0.9283
with ComplementNB,                   k-value: 5, accuracy: [0.69230769 0.69230769 0.76     0.64     0.56    ] 0.6689
with DecisionTreeClassifier,         k-value: 5, accuracy: [0.92307692 1.         0.96     0.96     0.92    ] 0.9526
with DummyClassifier,                k-value: 5, accuracy: [0.30769231 0.30769231 0.24     0.32     0.36    ] 0.3071
with ExtraTreeClassifier,            k-value: 5, accuracy: [0.88461538 1.         1.       0.92     0.92    ] 0.9449
with ExtraTreesClassifier,           k-value: 5, accuracy: [0.92307692 1.         0.96     0.96     0.88    ] 0.9446
with GaussianNB,                     k-value: 5, accuracy: [0.88461538 1.         0.96     0.96     0.92    ] 0.9449
with GaussianProcessClassifier,      k-value: 5, accuracy: [0.96153846 0.96153846 0.96     1.       0.88    ] 0.9526
with GradientBoostingClassifier,     k-value: 5, accuracy: [0.92307692 1.         0.96     0.96     0.84    ] 0.9366
with HistGradientBoostingClassifier, k-value: 5, accuracy: [0.92307692 1.         0.96     0.92     0.92    ] 0.9446
with KNeighborsClassifier,           k-value: 5, accuracy: [0.96153846 1.         0.96     0.96     0.88    ] 0.9523
with LabelPropagation,               k-value: 5, accuracy: [0.96153846 1.         0.96     0.96     0.92    ] 0.9603
with LabelSpreading,                 k-value: 5, accuracy: [0.96153846 1.         0.96     0.96     0.92    ] 0.9603
with LinearDiscriminantAnalysis,     k-value: 5, accuracy: [0.96153846 1.         1.       0.96     0.96    ] 0.9763
with LinearSVC,                      k-value: 5, accuracy: [0.88461538 1.         0.96     0.88     0.92    ] 0.9289
with LogisticRegression,             k-value: 5, accuracy: [0.96153846 1.         0.96     1.       0.88    ] 0.9603
with LogisticRegressionCV,           k-value: 5, accuracy: [0.96153846 1.         1.       1.       0.88    ] 0.9683
with MLPClassifier,                  k-value: 5, accuracy: [1.         1.         0.96     1.       0.88    ] 0.968
with MultinomialNB,                  k-value: 5, accuracy: [1.         0.92307692 0.96     0.8      0.64    ] 0.8646
with NearestCentroid,                k-value: 5, accuracy: [0.92307692 0.96153846 0.88     1.       0.84    ] 0.9209
with NuSVC,                          k-value: 5, accuracy: [0.96153846 0.96153846 0.92     1.       0.88    ] 0.9446
with PassiveAggressiveClassifier,    k-value: 5, accuracy: [0.84615385 0.88461538 0.76     0.8      0.56    ] 0.7702
with Perceptron,                     k-value: 5, accuracy: [0.30769231 0.69230769 0.84     0.68     0.56    ] 0.616
with QuadraticDiscriminantAnalysis,  k-value: 5, accuracy: [1.         1.         1.       0.96     0.92    ] 0.976
with RadiusNeighborsClassifier,      k-value: 5, accuracy: [       nan 0.96153846 0.92     1.       0.88    ] nan
with RandomForestClassifier,         k-value: 5, accuracy: [0.92307692 1.         0.96     0.92     0.88    ] 0.9366
with RidgeClassifier,                k-value: 5, accuracy: [0.80769231 0.96153846 0.84     0.72     0.84    ] 0.8338
with RidgeClassifierCV,              k-value: 5, accuracy: [0.80769231 0.96153846 0.84     0.72     0.84    ] 0.8338
with SGDClassifier,                  k-value: 5, accuracy: [0.88461538 0.69230769 0.8      0.76     0.88    ] 0.8034
with SVC,                            k-value: 5, accuracy: [0.96153846 0.96153846 0.92     1.       0.88    ] 0.9446

ClassifierChain will not work for this particular data-set
MultiOutputClassifier will not work for this particular data-set
OneVsOneClassifier will not work for this particular data-set
OneVsRestClassifier will not work for this particular data-set
OutputCodeClassifier will not work for this particular data-set
StackingClassifier will not work for this particular data-set
VotingClassifier will not work for this particular data-set

the number of working models: 34
the number of candidate models: 41
'''