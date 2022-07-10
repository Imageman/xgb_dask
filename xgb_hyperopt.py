# -*- coding: utf-8 -*-

'''

Подбор гиперпараметров для XGBoost с помощью hyperopt.

Справка по XGBoost:
http://xgboost.readthedocs.io/en/latest/

Справка по hyperopt:
https://github.com/hyperopt/hyperopt/wiki/FMin
http://fastml.com/optimizing-hyperparams-with-hyperopt/
https://conference.scipy.org/proceedings/scipy2013/pdfs/bergstra_hyperopt.pdf

'''
import sys

import xgboost
import sklearn
import numpy
import hyperopt
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import numpy as np


# кол-во случайных наборов гиперпараметров
from sklearn.metrics import accuracy_score
from loguru import logger

xgboost_func = xgboost.XGBRegressor

N_HYPEROPT_PROBES = 100

# алгоритм сэмплирования гиперпараметров
HYPEROPT_ALGO = tpe.suggest  #  tpe.suggest OR hyperopt.rand.suggest

# ----------------------------------------------------------
'''
(X_train, y_train, X_val, y_val, X_test, y_test ) =  mnist_vae.load_mnist()
'''

# ---------------------------------------------------------------------

def get_xgboost_model(space):
    _max_depth = int(space['max_depth'])
    _min_child_weight = space['min_child_weight']
    _subsample = space['subsample']
    _gamma = space['gamma'] if 'gamma' in space else 0.01
    _eta = space['eta']
    _seed = space['seed'] if 'seed' in space else 123456
    _colsample_bytree = space['colsample_bytree']
    _colsample_bylevel = space['colsample_bylevel']
    _n_estimators = int(space['n_estimators'])
    booster = space['booster'] if 'booster' in space else 'gbtree'

    model = xgboost_func(max_depth=_max_depth,
                                  min_child_weight=_min_child_weight,
                                  subsample=_subsample,
                                  gamma=_gamma,
                                  seed=_seed,
                                  colsample_bytree=_colsample_bytree,
                                  learning_rate=_eta,
                                  colsample_bylevel=_colsample_bylevel,
                                  n_estimators=_n_estimators)

    return model

# ---------------------------------------------------------------------

obj_call_count = 0
cur_best_loss = np.inf
log_writer = open( 'xgb-hyperopt-log.txt', 'w' )


def test(model, df_test_X, df_test_Y):
            # make predictions for test data
            y_pred = model.predict(df_test_X)
            predictions = [round(value) for value in y_pred]
            # evaluate predictions
            accuracy = accuracy_score(df_test_Y, predictions)
            from sklearn.metrics import f1_score
            f1 = f1_score(df_test_Y, predictions)
            logger.info("Accuracy: {:.2f}% F1: {:.2f}% ".format(accuracy * 100.0, f1*100))
            return f1

def objective(space):
    global obj_call_count, cur_best_loss

    obj_call_count += 1

    logger.debug('XGB objective call #{} cur_best_loss={:7.5f}'.format(obj_call_count,cur_best_loss) )

    model = get_xgboost_model(space)
    model.fit(X_train, y_train)

    #print(type(space))
    #sorted_params = sorted(space.iteritems(), key=lambda z: z[0])
    #params_str = str.join(' ', ['{}={}'.format(k, v) for k, v in sorted_params])
    params_str = str(space)
    logger.debug('Params: {}'.format(params_str) )

    #loss = model.best_score
    test_loss=1-test(model,  X_test, y_test)
    logger.debug('test_acc={}'.format(test_loss))

    log_writer.write('loss={:<7.5f} Params:{} \n'.format(test_loss, params_str ))
    log_writer.flush()

    if test_loss<cur_best_loss:
        cur_best_loss = test_loss
        logger.info('!!!!!!!! NEW BEST LOSS={}'.format(cur_best_loss))
        logger.info('Params: {}'.format(params_str))

    return {'loss':test_loss, 'status': STATUS_OK }


# --------------------------------------------------------------------------------

space ={
        #'booster': hp.choice( 'booster',  ['dart', 'gbtree'] ),
        'max_depth': hp.quniform("max_depth", 5, 9, 1),
        'n_estimators':hp.quniform("n_estimators", 50, 100, 1),
        'min_child_weight': hp.quniform ('min_child_weight', 1, 10, 1),
        'subsample': hp.uniform ('subsample', 0.7, 1.0),
        'gamma': hp.uniform('gamma', 1.0, 3.5),
        # 'gamma': hp.loguniform('gamma', -5.0, 0.0),
        'eta': hp.uniform('eta', 0.02, 0.045),
        #'eta': hp.loguniform('eta', -4.6, -2.3),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.50, 1.0),
        'colsample_bylevel': hp.uniform('colsample_bylevel', 0.80, 1.0),
        #'seed': hp.randint('seed', 2000000)
       }


def do_process(train_x, train_y, test_x, test_y):
    global X_train, y_train, X_test, y_test
    X_train = train_x
    y_train = train_y
    X_test = test_x
    y_test = test_y
    trials = Trials()
    best = hyperopt.fmin(fn=objective,
                         space=space,
                         algo=HYPEROPT_ALGO,
                         max_evals=N_HYPEROPT_PROBES,
                         trials=trials,
                         verbose=1)

    print('-'*50)
    logger.info('The best params:')
    logger.info( best )
    print('\n\n')
