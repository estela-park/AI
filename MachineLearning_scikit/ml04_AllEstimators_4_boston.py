import time
import warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import load_boston
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import all_estimators
from sklearn.metrics import r2_score


dataset = load_boston()

x = dataset.data   
# (506, 13)
y = dataset.target
# (506, )

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85)

max_abs_scaler = MaxAbsScaler()
max_abs_scaler.fit(x_train)
x_train_ma = max_abs_scaler.transform(x_train)
x_test_ma = max_abs_scaler.transform(x_test)

all_model = all_estimators('regressor')
count = 0
for model in all_model:
    try: 
        start = time.time()
        model_ma = model[1]()
        model_ma.fit(x_train_ma, y_train)
        predict_ma = model_ma.predict(x_test_ma)
        r2 = r2_score(y_test, predict_ma)
        end = time.time() - start
        count += 1

        print()
        print('+ model used:',model[0])
        print('it took',end)
        print('accuracy score:', r2)

    except:
        print(f'{model[0]} will not work for this particular data-set')
        continue

print('the number of working models:', count)
print('the number of candidate models:', len(all_model))

'''
+ model used: ARDRegression
it took 0.012930631637573242
accuracy score: 0.7451958270302761

+ model used: AdaBoostRegressor
it took 0.06485557556152344
accuracy score: 0.8726744717491504

+ model used: BaggingRegressor
it took 0.02393484115600586
accuracy score: 0.8861275741901062

+ model used: BayesianRidge
it took 0.002991199493408203
accuracy score: 0.7424109173111035

+ model used: CCA
it took 0.0020079612731933594
accuracy score: 0.707189663135632

+ model used: DecisionTreeRegressor
it took 0.0029785633087158203
accuracy score: 0.8302815948608308

+ model used: DummyRegressor
it took 0.000997781753540039
accuracy score: -0.015569972341648475

+ model used: ElasticNet
it took 0.0009970664978027344
accuracy score: 0.08980746502187809

+ model used: ElasticNetCV
it took 0.03889608383178711
accuracy score: 0.7320558453941943

+ model used: ExtraTreeRegressor
it took 0.0019943714141845703
accuracy score: 0.8665220179678298

+ model used: ExtraTreesRegressor
it took 0.13830971717834473
accuracy score: 0.924286544498776

+ model used: GammaRegressor
it took 0.0019948482513427734
accuracy score: 0.10561625898428684

+ model used: GaussianProcessRegressor
it took 0.008975982666015625
accuracy score: -0.17201510377509144

+ model used: GradientBoostingRegressor
it took 0.07496523857116699
accuracy score: 0.9192143132741047

+ model used: HistGradientBoostingRegressor
it took 0.20210838317871094
accuracy score: 0.9274083448955938

+ model used: HuberRegressor
it took 0.018949508666992188
accuracy score: 0.7253276499271903

IsotonicRegression will not work for this particular data-set

+ model used: KNeighborsRegressor
it took 0.000997304916381836
accuracy score: 0.6663065844077518

+ model used: KernelRidge
it took 0.0029921531677246094
accuracy score: 0.7050647336107667

+ model used: Lars
it took 0.0019943714141845703
accuracy score: 0.7437067057350001

+ model used: LarsCV
it took 0.011967658996582031
accuracy score: 0.7445897956346963

+ model used: Lasso
it took 0.0
accuracy score: 0.1982700240469032

+ model used: LassoCV
it took 0.04089021682739258
accuracy score: 0.7438061167232731

+ model used: LassoLars
it took 0.0009996891021728516
accuracy score: -0.015569972341648475

+ model used: LassoLarsCV
it took 0.012964248657226562
accuracy score: 0.7445606954632664

+ model used: LassoLarsIC
it took 0.001994609832763672
accuracy score: 0.7445033387310183

+ model used: LinearRegression
it took 0.0
accuracy score: 0.744090859786223

+ model used: LinearSVR
it took 0.0009975433349609375
accuracy score: 0.48667307950603444

+ model used: MLPRegressor
it took 0.25766754150390625
accuracy score: 0.3041975660931939

MultiOutputRegressor will not work for this particular data-set
MultiTaskElasticNet will not work for this particular data-set
MultiTaskElasticNetCV will not work for this particular data-set
MultiTaskLasso will not work for this particular data-set
MultiTaskLassoCV will not work for this particular data-set

+ model used: NuSVR
it took 0.008006095886230469
accuracy score: 0.5144447859810315

+ model used: OrthogonalMatchingPursuit
it took 0.0010271072387695312
accuracy score: 0.5402630858487868

+ model used: OrthogonalMatchingPursuitCV
it took 0.003961801528930664
accuracy score: 0.7237930303280042

+ model used: PLSCanonical
it took 0.000997781753540039
accuracy score: -1.6125816683940495

+ model used: PLSRegression
it took 0.0
accuracy score: 0.7196550083237012

+ model used: PassiveAggressiveRegressor
it took 0.000997304916381836
accuracy score: 0.7234140520033674

+ model used: PoissonRegressor
it took 0.003989219665527344
accuracy score: 0.47639279205289653

+ model used: RANSACRegressor
it took 0.03400087356567383
accuracy score: 0.6828338224419119

+ model used: RadiusNeighborsRegressor
it took 0.003989458084106445
accuracy score: 0.23581237055995163

+ model used: RandomForestRegressor
it took 0.22490477561950684
accuracy score: 0.898907589111064

RegressorChain will not work for this particular data-set

+ model used: Ridge
it took 0.0010254383087158203
accuracy score: 0.7209086554829057

+ model used: RidgeCV
it took 0.0009698867797851562
accuracy score: 0.7426446502027777

+ model used: SGDRegressor
it took 0.015956878662109375
accuracy score: 0.7019493207606814

+ model used: SVR
it took 0.011527538299560547
accuracy score: 0.5028718499706085

StackingRegressor will not work for this particular data-set

+ model used: TheilSenRegressor
it took 0.4802863597869873
accuracy score: 0.7088872857726912

+ model used: TransformedTargetRegressor
it took 0.0019571781158447266
accuracy score: 0.744090859786223

+ model used: TweedieRegressor
it took 0.0019943714141845703
accuracy score: 0.11155815262629387

VotingRegressor will not work for this particular data-set

the number of working models: 45
the number of candidate models: 54
'''