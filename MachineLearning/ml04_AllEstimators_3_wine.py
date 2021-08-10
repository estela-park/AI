import time
import warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import load_wine
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score


dataset = load_wine()

x = dataset.data   
y = dataset.target 

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
        model_ma.fit(x_train_ma, y_train)
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
+ model used: AdaBoostClassifier
it took 0.055817604064941406
accuracy score: 1.0

+ model used: BaggingClassifier
it took 0.013992786407470703
accuracy score: 1.0

+ model used: BernoulliNB
it took 0.000995635986328125
accuracy score: 0.37037037037037035

+ model used: CalibratedClassifierCV
it took 0.028952836990356445
accuracy score: 1.0

+ model used: CategoricalNB
it took 0.0019948482513427734
accuracy score: 0.4074074074074074

ClassifierChain will not work for this particular data-set

+ model used: ComplementNB
it took 0.0009975433349609375
accuracy score: 0.8888888888888888

+ model used: DecisionTreeClassifier
it took 0.0
accuracy score: 0.9629629629629629

+ model used: DummyClassifier
it took 0.0
accuracy score: 0.37037037037037035

+ model used: ExtraTreeClassifier
it took 0.0
accuracy score: 0.8518518518518519

+ model used: ExtraTreesClassifier
it took 0.07483053207397461
accuracy score: 1.0

+ model used: GaussianNB
it took 0.0
accuracy score: 1.0

+ model used: GaussianProcessClassifier
it took 0.030916929244995117
accuracy score: 1.0

+ model used: GradientBoostingClassifier
it took 0.20547842979431152
accuracy score: 0.9259259259259259

+ model used: HistGradientBoostingClassifier
it took 0.16655373573303223
accuracy score: 1.0

+ model used: KNeighborsClassifier
it took 0.001994609832763672
accuracy score: 0.9629629629629629

+ model used: LabelPropagation
it took 0.001995086669921875
accuracy score: 0.9629629629629629

+ model used: LabelSpreading
it took 0.000997304916381836
accuracy score: 0.9629629629629629

+ model used: LinearDiscriminantAnalysis
it took 0.0019941329956054688
accuracy score: 0.9629629629629629

+ model used: LinearSVC
it took 0.001994609832763672
accuracy score: 1.0

+ model used: LogisticRegression
it took 0.010970592498779297
accuracy score: 1.0

+ model used: LogisticRegressionCV
it took 0.510634183883667
accuracy score: 1.0

+ model used: MLPClassifier
it took 0.11970758438110352
accuracy score: 1.0

MultiOutputClassifier will not work for this particular data-set

+ model used: MultinomialNB
it took 0.0009968280792236328
accuracy score: 0.8518518518518519

+ model used: NearestCentroid
it took 0.0009970664978027344
accuracy score: 0.9629629629629629

+ model used: NuSVC
it took 0.0019948482513427734
accuracy score: 1.0

OneVsOneClassifier will not work for this particular data-set
OneVsRestClassifier will not work for this particular data-set
OutputCodeClassifier will not work for this particular data-set

+ model used: PassiveAggressiveClassifier
it took 0.0019958019256591797
accuracy score: 0.9259259259259259

+ model used: Perceptron
it took 0.0019943714141845703
accuracy score: 0.9629629629629629

+ model used: QuadraticDiscriminantAnalysis
it took 0.00099945068359375
accuracy score: 1.0

+ model used: RadiusNeighborsClassifier
it took 0.001994609832763672
accuracy score: 0.4444444444444444

+ model used: RandomForestClassifier
it took 0.09776639938354492
accuracy score: 1.0

+ model used: RidgeClassifier
it took 0.002024412155151367
accuracy score: 1.0

+ model used: RidgeClassifierCV
it took 0.001995563507080078
accuracy score: 0.9629629629629629

+ model used: SGDClassifier
it took 0.000997304916381836
accuracy score: 1.0

+ model used: SVC
it took 0.0009980201721191406
accuracy score: 1.0

StackingClassifier will not work for this particular data-set
VotingClassifier will not work for this particular data-set

the number of working models: 34
the number of candidate models: 41
'''