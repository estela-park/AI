import time
import warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import all_estimators
from sklearn.metrics import r2_score


dataset = load_diabetes()

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
it took 0.012964010238647461
accuracy score: 0.64112156496889

+ model used: AdaBoostRegressor
it took 0.06183314323425293
accuracy score: 0.5410118481447372

+ model used: BaggingRegressor
it took 0.01994633674621582
accuracy score: 0.6403239135472234

+ model used: BayesianRidge
it took 0.0019943714141845703
accuracy score: 0.6450635957451032

+ model used: CCA
it took 0.0019941329956054688
accuracy score: 0.6338736044158246

+ model used: DecisionTreeRegressor
it took 0.001995086669921875
accuracy score: 0.15655952650042226

+ model used: DummyRegressor
it took 0.0
accuracy score: -0.005836384496178404

+ model used: ElasticNet
it took 0.000997304916381836
accuracy score: 0.3014745010082026

+ model used: ElasticNetCV
it took 0.036901235580444336
accuracy score: 0.6309115778509078

+ model used: ExtraTreeRegressor
it took 0.0009970664978027344
accuracy score: 0.0030943416850399696

+ model used: ExtraTreesRegressor
it took 0.12167501449584961
accuracy score: 0.6215963890573308

+ model used: GammaRegressor
it took 0.0019941329956054688
accuracy score: 0.19724514685041983

+ model used: GaussianProcessRegressor
it took 0.00797891616821289
accuracy score: -5.502818284650581

+ model used: GradientBoostingRegressor
it took 0.05684828758239746
accuracy score: 0.5851286399604925

+ model used: HistGradientBoostingRegressor
it took 0.17553043365478516
accuracy score: 0.600809203517445

+ model used: HuberRegressor
it took 0.008976221084594727
accuracy score: 0.6491443562867154

IsotonicRegression will not work for this particular data-set

+ model used: KNeighborsRegressor
it took 0.0019948482513427734
accuracy score: 0.46507561784705065

+ model used: KernelRidge
it took 0.0029916763305664062
accuracy score: -3.661741773277962

+ model used: Lars
it took 0.0019943714141845703
accuracy score: 0.6553902582977043

+ model used: LarsCV
it took 0.0109710693359375
accuracy score: 0.6351990143618329

+ model used: Lasso
it took 0.0
accuracy score: 0.6121531660982341

+ model used: LassoCV
it took 0.04288530349731445
accuracy score: 0.6456343974410742

+ model used: LassoLars
it took 0.000997781753540039
accuracy score: 0.4180550936117169

+ model used: LassoLarsCV
it took 0.01196742057800293
accuracy score: 0.6453124471491053

+ model used: LassoLarsIC
it took 0.0019953250885009766
accuracy score: 0.6432933976972177

+ model used: LinearRegression
it took 0.0
accuracy score: 0.6553902582977046

+ model used: LinearSVR
it took 0.000997781753540039
accuracy score: 0.10846708656111004

+ model used: MLPRegressor
it took 0.25730133056640625
accuracy score: -2.003100959576632

MultiOutputRegressor will not work for this particular data-set
MultiTaskElasticNet will not work for this particular data-set
MultiTaskElasticNetCV will not work for this particular data-set
MultiTaskLasso will not work for this particular data-set
MultiTaskLassoCV will not work for this particular data-set

+ model used: NuSVR
it took 0.0059661865234375
accuracy score: 0.134860730793962

+ model used: OrthogonalMatchingPursuit
it took 0.0009720325469970703
accuracy score: 0.2391784204322207

+ model used: OrthogonalMatchingPursuitCV
it took 0.004987955093383789
accuracy score: 0.6405606109637472

+ model used: PLSCanonical
it took 0.0009968280792236328
accuracy score: -1.0362257591586417

+ model used: PLSRegression
it took 0.0009732246398925781
accuracy score: 0.6606377035181015

+ model used: PassiveAggressiveRegressor
it took 0.0
accuracy score: 0.6413592389727383

+ model used: PoissonRegressor
it took 0.0029859542846679688
accuracy score: 0.6273361684678497

+ model used: RANSACRegressor
it took 0.03496503829956055
accuracy score: 0.21847015105182066

RadiusNeighborsRegressor will not work for this particular data-set

+ model used: RandomForestRegressor
it took 0.18949317932128906
accuracy score: 0.6119670495594736

RegressorChain will not work for this particular data-set

+ model used: Ridge
it took 0.0009975433349609375
accuracy score: 0.648011586474466

+ model used: RidgeCV
it took 0.0009975433349609375
accuracy score: 0.6480115864744602

+ model used: SGDRegressor
it took 0.003988981246948242
accuracy score: 0.6478707727471369

+ model used: SVR
it took 0.007978200912475586
accuracy score: 0.1310786047652347

StackingRegressor will not work for this particular data-set

+ model used: TheilSenRegressor
it took 0.42098236083984375
accuracy score: 0.6198645828377609

+ model used: TransformedTargetRegressor
it took 0.0009963512420654297
accuracy score: 0.6553902582977046

+ model used: TweedieRegressor
it took 0.0019943714141845703
accuracy score: 0.207511062762186

VotingRegressor will not work for this particular data-set

the number of working models: 44
the number of candidate models: 54
'''