import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.utils import all_estimators


# 1. Data-prep
dataset = load_breast_cancer()

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
with AdaBoostClassifier,             k-value: 5, accuracy: [0.95876289 0.96907216 0.94845361 0.9375    0.96875  ] 0.9565
with BaggingClassifier,              k-value: 5, accuracy: [0.97938144 0.95876289 0.95876289 0.9270833 0.96875  ] 0.9585
with BernoulliNB,                    k-value: 5, accuracy: [0.58762887 0.63917526 0.57731959 0.6875    0.6770833] 0.6337
with CalibratedClassifierCV,         k-value: 5, accuracy: [0.90721649 0.91752577 0.92783505 0.9166666 0.9479166] 0.9234
with CategoricalNB,                  k-value: 5, accuracy: [0.90721649        nan        nan       nan       nan] nan
with ComplementNB,                   k-value: 5, accuracy: [0.88659794 0.90721649 0.86597938 0.9166666 0.9270833] 0.9007
with DecisionTreeClassifier,         k-value: 5, accuracy: [0.94845361 0.93814433 0.93814433 0.8333333 0.9270833] 0.917
with DummyClassifier,                k-value: 5, accuracy: [0.58762887 0.63917526 0.57731959 0.6875    0.6770833] 0.6337
with ExtraTreeClassifier,            k-value: 5, accuracy: [0.92783505 0.93814433 0.91752577 0.8958333 0.9479166] 0.9255
with ExtraTreesClassifier,           k-value: 5, accuracy: [0.96907216 0.96907216 0.95876289 0.9479166 0.9895833] 0.9669
with GaussianNB,                     k-value: 5, accuracy: [0.95876289 0.93814433 0.93814433 0.9270833 0.9791666] 0.9483
with GaussianProcessClassifier,      k-value: 5, accuracy: [0.90721649 0.92783505 0.92783505 0.90625   0.9375   ] 0.9213
with GradientBoostingClassifier,     k-value: 5, accuracy: [0.96907216 0.95876289 0.95876289 0.9270833 0.9895833] 0.9607
with HistGradientBoostingClassifier, k-value: 5, accuracy: [0.97938144 0.96907216 0.95876289 0.9375    0.96875  ] 0.9627
with KNeighborsClassifier,           k-value: 5, accuracy: [0.90721649 0.91752577 0.93814433 0.90625   0.9583333] 0.9255
with LabelPropagation,               k-value: 5, accuracy: [0.43298969 0.3814433  0.43298969 0.3229166 0.34375  ] 0.3828
with LabelSpreading,                 k-value: 5, accuracy: [0.43298969 0.3814433  0.43298969 0.3229166 0.34375  ] 0.3828
with LinearDiscriminantAnalysis,     k-value: 5, accuracy: [0.96907216 0.96907216 0.94845361 0.8958333 0.9895833] 0.9544
with LinearSVC,                      k-value: 5, accuracy: [0.92783505 0.83505155 0.94845361 0.8854166 0.9375   ] 0.9069
with LogisticRegression,             k-value: 5, accuracy: [0.96907216 0.94845361 0.93814433 0.9270833 0.9791666] 0.9524
with LogisticRegressionCV,           k-value: 5, accuracy: [0.96907216 0.95876289 0.94845361 0.9166666 0.9895833] 0.9565
with MLPClassifier,                  k-value: 5, accuracy: [0.91752577 0.93814433 0.93814433 0.90625   0.9583333] 0.9317
with MultinomialNB,                  k-value: 5, accuracy: [0.88659794 0.89690722 0.87628866 0.9166666 0.9166666] 0.8986
with NearestCentroid,                k-value: 5, accuracy: [0.87628866 0.88659794 0.88659794 0.875     0.9583333] 0.8966
with NuSVC,                          k-value: 5, accuracy: [0.82474227 0.88659794 0.84536082 0.875     0.9583333] 0.878
with PassiveAggressiveClassifier,    k-value: 5, accuracy: [0.91752577 0.88659794 0.91752577 0.8958333 0.7291666] 0.8693
with Perceptron,                     k-value: 5, accuracy: [0.64948454 0.81443299 0.92783505 0.8020833 0.8854166] 0.8159
with QuadraticDiscriminantAnalysis,  k-value: 5, accuracy: [0.96907216 0.96907216 0.95876289 0.90625   0.96875  ] 0.9544
with RadiusNeighborsClassifier,      k-value: 5, accuracy: [nan        nan        nan        nan       nan      ] nan
with RandomForestClassifier,         k-value: 5, accuracy: [0.95876289 0.96907216 0.94845361 0.9479166 0.9895833] 0.9628
with RidgeClassifier,                k-value: 5, accuracy: [0.96907216 0.95876289 0.92783505 0.9166666 0.9791666] 0.9503
with RidgeClassifierCV,              k-value: 5, accuracy: [0.96907216 0.95876289 0.91752577 0.90625   0.9791666] 0.9462
with SGDClassifier,                  k-value: 5, accuracy: [0.87628866 0.91752577 0.91752577 0.8645833 0.8854166] 0.8923
with SVC,                            k-value: 5, accuracy: [0.88659794 0.88659794 0.89690722 0.8958333 0.9583333] 0.9049

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