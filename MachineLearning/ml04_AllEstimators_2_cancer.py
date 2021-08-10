import time
import warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score


dataset = load_breast_cancer()

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
it took 0.0987083911895752
accuracy score: 0.9767441860465116

+ model used: BaggingClassifier
it took 0.03693079948425293
accuracy score: 0.9302325581395349

+ model used: BernoulliNB
it took 0.0010225772857666016
accuracy score: 0.6627906976744186

+ model used: CalibratedClassifierCV
it took 0.021947145462036133
accuracy score: 0.9534883720930233

+ model used: CategoricalNB
it took 0.003981113433837891
accuracy score: 0.6976744186046512

ClassifierChain will not work for this particular data-set

+ model used: ComplementNB
it took 0.0
accuracy score: 0.9069767441860465

+ model used: DecisionTreeClassifier
it took 0.004986763000488281
accuracy score: 0.9418604651162791

+ model used: DummyClassifier
it took 0.0
accuracy score: 0.6627906976744186

+ model used: ExtraTreeClassifier
it took 0.0
accuracy score: 0.9302325581395349

+ model used: ExtraTreesClassifier
it took 0.08679389953613281
accuracy score: 0.9651162790697675

+ model used: GaussianNB
it took 0.0010247230529785156
accuracy score: 0.9418604651162791

+ model used: GaussianProcessClassifier
it took 0.07562708854675293
accuracy score: 0.9651162790697675

+ model used: GradientBoostingClassifier
it took 0.2593381404876709
accuracy score: 0.9767441860465116

+ model used: HistGradientBoostingClassifier
it took 0.2822449207305908
accuracy score: 0.9767441860465116

+ model used: KNeighborsClassifier
it took 0.003988981246948242
accuracy score: 0.9651162790697675

+ model used: LabelPropagation
it took 0.004987001419067383
accuracy score: 0.9651162790697675

+ model used: LabelSpreading
it took 0.005984306335449219
accuracy score: 0.9651162790697675

+ model used: LinearDiscriminantAnalysis
it took 0.004986286163330078
accuracy score: 0.9651162790697675

+ model used: LinearSVC
it took 0.0029921531677246094
accuracy score: 0.9534883720930233

+ model used: LogisticRegression
it took 0.005984067916870117
accuracy score: 0.9651162790697675

+ model used: LogisticRegressionCV
it took 0.49114179611206055
accuracy score: 0.9534883720930233

+ model used: MLPClassifier
it took 0.5804047584533691
accuracy score: 0.9534883720930233

MultiOutputClassifier will not work for this particular data-set

+ model used: MultinomialNB
it took 0.0010352134704589844
accuracy score: 0.872093023255814

+ model used: NearestCentroid
it took 0.0
accuracy score: 0.9302325581395349

+ model used: NuSVC
it took 0.009002685546875
accuracy score: 0.9534883720930233

OneVsOneClassifier will not work for this particular data-set
OneVsRestClassifier will not work for this particular data-set
OutputCodeClassifier will not work for this particular data-set

+ model used: PassiveAggressiveClassifier
it took 0.0019943714141845703
accuracy score: 0.9534883720930233

+ model used: Perceptron
it took 0.000997304916381836
accuracy score: 0.9651162790697675

+ model used: QuadraticDiscriminantAnalysis
it took 0.003989458084106445
accuracy score: 0.9534883720930233

RadiusNeighborsClassifier will not work for this particular data-set

+ model used: RandomForestClassifier
it took 0.13366961479187012
accuracy score: 0.9767441860465116

+ model used: RidgeClassifier
it took 0.0029916763305664062
accuracy score: 0.9534883720930233

+ model used: RidgeClassifierCV
it took 0.002991914749145508
accuracy score: 0.9534883720930233

+ model used: SGDClassifier
it took 0.0009970664978027344
accuracy score: 0.9302325581395349

+ model used: SVC
it took 0.002991914749145508
accuracy score: 0.9534883720930233

StackingClassifier will not work for this particular data-set
VotingClassifier will not work for this particular data-set

the number of working models: 33
the number of candidate models: 41
'''