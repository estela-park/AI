import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import load_iris
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score

'''
sklearn.utils.all_estimators(type_filter=None)
  -it returns a list of all estimators from sklearn.

This function crawls the module and gets all classes that inherit from BaseEstimator. 
Classes that are defined in test-modules are not included.

Parameters
  type_filter{“classifier”, “regressor”, “cluster”, “transformer”} or list of such str, default=None
  Which kind of estimators should be returned. 
  If None, no filter is applied and all estimators are returned. 
  Possible values are ‘classifier’, ‘regressor’, ‘cluster’ and ‘transformer’,
  or a list of these to get the estimators that fit at least one of the types.

Returns
  estimatorslist of tuples
  List of (name, class), where name is the class name as string and class is the actuall type of the class.
'''
dataset = load_iris()

x = dataset.data   
# (150, 4)
y = dataset.target 
# (150, )

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85) 

max_abs_scaler = MaxAbsScaler()
max_abs_scaler.fit(x_train)
x_train_ma = max_abs_scaler.transform(x_train)
x_test_ma = max_abs_scaler.transform(x_test)

all_model = all_estimators('classifier')
count = 0
for model in all_model:
    try: 
        start = time.time()
        model_ma = model[1]()
        # except data, most parameters are set default
        model_ma.fit(x_train_ma, y_train)

        # score, not every model provides score method.
        predict_ma = model_ma.predict(x_test_ma)
        acc = accuracy_score(y_test, predict_ma)
        end = time.time() - start
        count += 1

        print()
        print('+ model used:',model[0])
        print('it took',end)
        print('accuracy score:', acc)

    except:
        print(f'{model[0]} will not work for this particular data-set')
        continue

print('the number of working models:', count)
print('the number of candidate models:', len(all_model))

'''
    TypeError: __init__() missing 1 required positional argument: 'base_estimator'
****Use try-catch 

+ model used: AdaBoostClassifier
it took 0.04983663558959961
accuracy score: 0.9130434782608695

+ model used: BaggingClassifier
it took 0.012951374053955078
accuracy score: 0.9565217391304348

+ model used: BernoulliNB
it took 0.0010149478912353516
accuracy score: 0.2608695652173913

+ model used: CalibratedClassifierCV
it took 0.023936033248901367
accuracy score: 0.8695652173913043

+ model used: CategoricalNB
it took 0.0010256767272949219
accuracy score: 0.34782608695652173

ClassifierChain will not work for this particular data-set

+ model used: ComplementNB
it took 0.001024484634399414
accuracy score: 0.6956521739130435

+ model used: DecisionTreeClassifier
it took 0.0009706020355224609
accuracy score: 0.9130434782608695

+ model used: DummyClassifier
it took 0.0
accuracy score: 0.2608695652173913

+ model used: ExtraTreeClassifier
it took 0.0
accuracy score: 0.8695652173913043

+ model used: ExtraTreesClassifier
it took 0.071807861328125
accuracy score: 0.9565217391304348

+ model used: GaussianNB
it took 0.0009970664978027344
accuracy score: 0.9565217391304348

+ model used: GaussianProcessClassifier
it took 0.043906450271606445
accuracy score: 0.9565217391304348

+ model used: GradientBoostingClassifier
it took 0.13862919807434082
accuracy score: 0.9130434782608695

+ model used: HistGradientBoostingClassifier
it took 0.13164734840393066
accuracy score: 0.9565217391304348

+ model used: KNeighborsClassifier
it took 0.0019948482513427734
accuracy score: 0.9565217391304348

+ model used: LabelPropagation
it took 0.0009975433349609375
accuracy score: 0.9565217391304348

+ model used: LabelSpreading
it took 0.0019943714141845703
accuracy score: 0.9565217391304348

+ model used: LinearDiscriminantAnalysis
it took 0.0009970664978027344
accuracy score: 1.0

+ model used: LinearSVC
it took 0.000997304916381836
accuracy score: 0.9130434782608695

+ model used: LogisticRegression
it took 0.006981372833251953
accuracy score: 0.9565217391304348

+ model used: LogisticRegressionCV
it took 0.2872316837310791
accuracy score: 1.0

+ model used: MLPClassifier
it took 0.10873675346374512
accuracy score: 0.9565217391304348

MultiOutputClassifier will not work for this particular data-set

+ model used: MultinomialNB
it took 0.0009970664978027344
accuracy score: 0.8260869565217391

+ model used: NearestCentroid
it took 0.000997304916381836
accuracy score: 0.9565217391304348

+ model used: NuSVC
it took 0.001994609832763672
accuracy score: 0.9565217391304348

OneVsOneClassifier will not work for this particular data-set
OneVsRestClassifier will not work for this particular data-set
OutputCodeClassifier will not work for this particular data-set

+ model used: PassiveAggressiveClassifier
it took 0.003989458084106445
accuracy score: 0.8260869565217391

+ model used: Perceptron
it took 0.0009970664978027344
accuracy score: 0.782608695652174

+ model used: QuadraticDiscriminantAnalysis
it took 0.0032122135162353516
accuracy score: 0.9565217391304348

+ model used: RadiusNeighborsClassifier
it took 0.0009992122650146484
accuracy score: 0.2608695652173913

+ model used: RandomForestClassifier
it took 0.09377670288085938
accuracy score: 0.9565217391304348

+ model used: RidgeClassifier
it took 0.002721071243286133
accuracy score: 0.9130434782608695

+ model used: RidgeClassifierCV
it took 0.000997781753540039
accuracy score: 0.782608695652174

+ model used: SGDClassifier
it took 0.0019943714141845703
accuracy score: 0.9130434782608695

+ model used: SVC
it took 0.0009980201721191406
accuracy score: 0.9565217391304348

StackingClassifier will not work for this particular data-set
VotingClassifier will not work for this particular data-set

the number of candidate models: 41
the number of working models: 34
'''