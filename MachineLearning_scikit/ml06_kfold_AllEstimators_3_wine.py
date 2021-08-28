import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.utils import all_estimators


# 1. Data-prep
dataset = load_wine()

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
with AdaBoostClassifier,             k-value: 5, accuracy: [0.96774194 0.86666667 0.9       0.9       0.9      ] 0.9069
with BaggingClassifier,              k-value: 5, accuracy: [0.90322581 0.93333333 1.        0.9666666 0.9      ] 0.9406
with BernoulliNB,                    k-value: 5, accuracy: [0.35483871 0.4        0.4333333 0.3333333 0.5333333] 0.411
with CalibratedClassifierCV,         k-value: 5, accuracy: [0.90322581 0.96666667 0.9666666 0.8       0.9666666] 0.9206
with CategoricalNB,                  k-value: 5, accuracy: [       nan        nan       nan 0.9666666       nan] nan
with ComplementNB,                   k-value: 5, accuracy: [0.77419355 0.53333333 0.7666666 0.4666666 0.7      ] 0.6482
with DecisionTreeClassifier,         k-value: 5, accuracy: [0.93548387 0.93333333 0.9666666 1.        0.9333333] 0.9538
with DummyClassifier,                k-value: 5, accuracy: [0.35483871 0.4        0.4333333 0.3333333 0.5333333] 0.411
with ExtraTreeClassifier,            k-value: 5, accuracy: [0.96774194 0.93333333 0.8666666 0.9333333 0.8333333] 0.9069
with ExtraTreesClassifier,           k-value: 5, accuracy: [1.         1.         0.9666666 1.        0.9333333] 0.98
with GaussianNB,                     k-value: 5, accuracy: [0.90322581 1.         0.9666666 0.9666666 0.9333333] 0.954
with GaussianProcessClassifier,      k-value: 5, accuracy: [0.38709677 0.33333333 0.5333333 0.5666666 0.5      ] 0.4641
with GradientBoostingClassifier,     k-value: 5, accuracy: [0.90322581 0.96666667 0.9       0.8666666 0.9666666] 0.9206
with HistGradientBoostingClassifier, k-value: 5, accuracy: [0.96774194 0.96666667 1.        0.9666666 0.9666666] 0.9735
with KNeighborsClassifier,           k-value: 5, accuracy: [0.83870968 0.66666667 0.7333333 0.7       0.7      ] 0.7277
with LabelPropagation,               k-value: 5, accuracy: [0.58064516 0.33333333 0.5333333 0.5333333 0.3      ] 0.4561
with LabelSpreading,                 k-value: 5, accuracy: [0.58064516 0.33333333 0.5333333 0.5333333 0.3      ] 0.4561
with LinearDiscriminantAnalysis,     k-value: 5, accuracy: [1.         1.         1.        1.        0.9      ] 0.98
with LinearSVC,                      k-value: 5, accuracy: [0.87096774 0.8        0.9666666 0.7       0.9666666] 0.8609
with LogisticRegression,             k-value: 5, accuracy: [0.96774194 1.         0.9666666 0.8666666 0.9666666] 0.9535
with LogisticRegressionCV,           k-value: 5, accuracy: [0.90322581 1.         1.        0.9333333 0.9      ] 0.9473
with MLPClassifier,                  k-value: 5, accuracy: [0.12903226 0.43333333 0.6       0.8666666 0.2666666] 0.4591
with MultinomialNB,                  k-value: 5, accuracy: [0.83870968 0.76666667 0.8666666 0.8       0.9333333] 0.8411
with NearestCentroid,                k-value: 5, accuracy: [0.77419355 0.63333333 0.8       0.7       0.7      ] 0.7215
with NuSVC,                          k-value: 5, accuracy: [0.90322581 0.83333333 0.9666666 0.8333333 0.8666666] 0.8806
with PassiveAggressiveClassifier,    k-value: 5, accuracy: [0.70967742 0.36666667 0.7       0.4333333 0.5333333] 0.5486
with Perceptron,                     k-value: 5, accuracy: [0.70967742 0.5        0.7333333 0.4333333 0.6333333] 0.6019
with QuadraticDiscriminantAnalysis,  k-value: 5, accuracy: [1.         1.         1.        1.        1.       ] 1.0
with RadiusNeighborsClassifier,      k-value: 5, accuracy: [nan        nan        nan       nan       nan      ] nan
with RandomForestClassifier,         k-value: 5, accuracy: [0.96774194 1.         0.9666666 1.        0.9666666] 0.9802
with RidgeClassifier,                k-value: 5, accuracy: [1.         1.         0.9333333 1.        0.9333333] 0.9733
with RidgeClassifierCV,              k-value: 5, accuracy: [1.         1.         0.9333333 1.        0.9333333] 0.9733
with SGDClassifier,                  k-value: 5, accuracy: [0.74193548 0.56666667 0.6       0.4666666 0.7      ] 0.6151
with SVC,                            k-value: 5, accuracy: [0.77419355 0.56666667 0.8       0.6       0.6666666] 0.6815

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