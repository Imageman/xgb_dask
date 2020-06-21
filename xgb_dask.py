'''
почти оптимальный скрипт по обучению XGB
на вход подается CSV, а лучше Parquiet
входящие данные предобрабатываются (переименовываются колонки и удаляются ненужные колонки)
данные перемешиваются и делятся на обучающую и тестовую выборки (выборки записываются в паркет)
выполняются несколько иттераций с автоулучшениеим.
На каждой иттерации берется только часть данных, что бы влезало в память
'''
import sys
from os import path
from datetime import datetime
from random import random, randint
import re

import numpy
# import treelite as treelite
import xgboost
import pandas
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from xgboost import plot_importance
import dask.dataframe as dd
import logging

import fastparquet
# import snappy

import exportXG

subsample = 0.99
max_depth = 11
n_estimators = 10
max_datasize = 210000

# create logger with 'xgb_dask'
logger = logging.getLogger('xgb_dask')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('xgb_dask.log')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(message)s - %(name)s - %(levelname)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

from psutil import virtual_memory

mem = virtual_memory()
if mem.total < 16e09:
    max_datasize = round(max_datasize / (17e09 / mem.total))
    logger.info('max_datasize={}'.format(max_datasize))

import m2cgen

def exportC(model, filename):
    code = m2cgen.export_to_c_sharp(model)
    text_file = open(filename, "w")
    n = text_file.write(code)
    text_file.close()
    return


def shuffle_dataframe(df, n=1, axis=0):
    # very slow?
    df = df.copy()
    for _ in range(n):
        df.apply(numpy.random.shuffle, axis=axis)
    return df


# A parameter grid for XGBoost
params = {
    'min_child_weight': [1, 5, 10],
    'gamma': [0.5, 1, 1.5, 2, 5],
    'subsample': [0.95, 0.98, 1.0],
    'colsample_bytree': [0.9, 0.95, 1.0],
    'learning_rate': [0.05, 0.1, 0.2, 0.3],
    'max_depth': [8, 9, 10, 11],
    'n_estimators': [190],
}


def getHyperParams(xgb, X, Y):
    # https://www.kaggle.com/tilii7/hyperparameter-grid-search-with-xgboost
    folds = 3
    param_comb = 20

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)

    random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='roc_auc',
                                       cv=skf.split(X, Y), verbose=3)  # , random_state=1001

    # Here we go
    start_time = timer(None)  # timing starts from this point for "start_time" variable
    random_search.fit(X, Y)
    timer(start_time)  # timing ends here for "start_time" variable
    print('\n Best estimator:')
    print(random_search.best_estimator_)
    print('\n Best hyperparameters:')
    print(random_search.best_params_)
    results = pandas.DataFrame(random_search.cv_results_)
    results.to_csv('xgb-random-grid-search-results-01.csv', index=False)


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        logger.debug('Time mark of begin')
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        # print('Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))
        logger.info('Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))
        return datetime.now()


def test(model, df_test_X, df_test_Y, save=True):
    global bestF1
    try:
        model.get_booster().dump_model('./result/anom_x_{}.dmp'.format('tmp'))
        # print("Accuracy:", end='', flush=True)
        # make predictions for test data
        y_pred = model.predict(df_test_X)
        predictions = [round(value) for value in y_pred]
        # evaluate predictions
        accuracy = accuracy_score(df_test_Y, predictions)
        from sklearn.metrics import f1_score
        f1 = f1_score(df_test_Y, predictions)
        logger.info("Accuracy: {:.2f}% F1: {:.2f}% ".format(accuracy * 100.0, f1 * 100))
    except:
        print('Error test, ignoring....')
        logger.error('Error test:' + sys.exc_info()[0])
        bestF1 = -1
    if (save and f1 > bestF1):
        model.save_model('./result/anom_x_{}.bst'.format(f1))
        model.get_booster().dump_model('./result/anom_x_{}.dmp'.format(f1))
        # exportC(model, './result/anom_x_{}.CS'.format(F1) )
        bestF1 = f1
        '''
            try:
                ExportC(model, 'xgb_class_{}.CS'.format(F1))
            except:
                print('Error export' )
            '''
    return f1


def colRenamer(x):
    # ищем запрещенные символы в названии и заменяем на безопасные
    t = re.sub('Result', 'result', x)
    t = re.sub('\s+', '', t)  # убираем пробелы
    t = re.sub('\[', 'T_', t)
    t = re.sub('\]', '_T', t)
    t = re.sub('\<', '_less', t)
    t = re.sub('\>', '_great', t)
    return t


def reoptimize_to_Parquiet(bigDF, outfilename):
    if bigDF is None:
        logger.info('Dataframe is None, read ' + outfilename)
    else:
        logger.debug('Start write ' + outfilename)
        bigDF.to_parquet(outfilename, engine='fastparquet')
        logger.info('Fin    write ' + outfilename)
    return dd.read_parquet(outfilename, engine='fastparquet')


def resampleTrainData(df_train):
    logger.debug('resampleData()')
    # из большой кучи df_train делаем случайную подвыборку
    # sample_ratio должен быть небольшой, что бы результат влез в память XGB
    # global df_train_Y
    # global df_train_X
    global sample_ratio
    df_train_X = df_train.sample(frac=sample_ratio).compute()
    df_train_Y = df_train_X['result']
    df_train_X.drop(columns=['result'], inplace=True)
    return df_train_X, df_train_Y


def preprocessRAWdata(filename):
    # загружаем данные, удаляем ненужное, разбиваем на обучение и тест
    global df_train, df_test
    logger.info('Load data ' + filename)
    if filename.find('.par') > 0:
        dataf = dd.read_parquet(filename, engine='fastparquet')
    else:
        dataf = dd.read_csv(filename, skiprows=[1], skipinitialspace=True)

    dataf = dataf.rename(columns=lambda x: colRenamer(x))  # меняем заголовок

    # теперь составим список записей для удаления
    cols_name = []
    for col in dataf.columns:
        '''
        match = re.match(r'^(smart_\d+)_', col)
        if match is not None and match.group(1) in self.cols_to_drop:
            cols_name.append(col)
        '''
        # колонки, которые нужно игнорировать по типу CalculateDTWdistance(
        if col.find('(') > 0:
            cols_name.append(col)
    logger.debug('Delete cols ' + str(cols_name).strip('[]'))
    dataf = dataf.drop(cols_name, axis=1)

    '''
    cols_name = []
    for col in dataf.columns:
        # колонки, которым нужно переделать тип
        if col.find('A')>-1:
            dataf[col]=dataf[col].astype(numpy.float32)
            cols_name.append(col)
    logger.debug('Float32 to cols ' + str(cols_name).strip('[]'))
    '''
    # dataf = dataf.drop(labels=lambda x: x.find('1'), axis=1)
    # print(dataf.head())

    logger.debug('Split data')
    # train_size = round( 0.9 * dataf.shape[0].compute())

    train_percent = 5 / 100  # для теста берем маленькую часть
    df_train, df_test = dataf.random_split([1 - train_percent, train_percent])
    return df_train, df_test


if __name__ == '__main__':
    # execute this only when run directly, not when imported!
    # Это условие нужно, что бы в Windows работал num_workers=x

    # print('start')
    logger.info('Start')
    start_timer = timer()
    # print('Old sys.getrecursionlimit={}'.format(sys.getrecursionlimit()))
    sys.setrecursionlimit(5000)
    # print('New sys.getrecursionlimit={}'.format(sys.getrecursionlimit()))
    logger.debug('New sys.getrecursionlimit={}'.format(sys.getrecursionlimit()))
    bestF1 = 0

    df_test = None
    df_train = None

    filename = 'small1_55.csv'
    filename = 'small.csv'
    filename = 'd:/tmp/StatSimil/JpgSimilarStat.csv'
    filename = 'test.csv'
    filename = 'small1_55.parquet'
    filename = 'JpgSimilarStat.parquet'

    if not path.exists('./data/test.parquet'):  # ! если есть "наш" паркет, то перескакиваем
        df_train, df_test = preprocessRAWdata(
            filename)  # !включаем это, ТОЛЬКО если хотим заново загрузить сырые данные!

    df_train = reoptimize_to_Parquiet(df_train, 'data/train.parquet')  # реоптимизация и сохранение или загрузка  data
    df_test = reoptimize_to_Parquiet(df_test, 'data/test.parquet')  # реоптимизация и сохранение или загрузка  data

    logger.debug('Test data process')
    df_test_Y = df_test['result']
    df_test = df_test.drop('result', axis=1)

    df_train['AT_0_T'] = 0  # вроде это неудачная фича (только не удалять, иначе сместятся индексы колонок)
    logger.debug('Get length df_train')
    total_data = df_train.shape[0].compute()  # len(df_train)
    sample_ratio = max_datasize / total_data
    if sample_ratio > 1:
        sample_ratio = 1

    '''
    logger.info('Start HyperParams optimize')
    sample_ratio = sample_ratio*0.95
    df_train_X, df_train_Y = resampleTrainData(df_train)  # сделали подвыборку
    # model = xgboost.XGBClassifier(max_depth=9, n_estimators=150, subsample=1)
    # getHyperParams(model, df_train_X, df_train_Y)
    import xgb_hyperopt
    xgb_hyperopt.do_process(df_train_X, df_train_Y, df_test.compute(), df_test_Y.compute())
    input("Press Enter to continue...")
    '''

    logger.info(
        'Total train data {}; test data {}.  Ratio of train part {:.3f}'.format(total_data, len(df_test), sample_ratio))

    start_timer: datetime = timer(start_timer)
    logger.info('xgboost train')
    model = None

    if path.exists('./1.bst.tmp'):
        logger.info('Try load ./1.bst.tmp')
        model = xgboost.XGBClassifier(max_depth=max_depth, n_estimators=n_estimators, subsample=subsample)
        model.load_model('./1.bst.tmp')
        f1score = test(model, df_test, df_test_Y, save=False)
        # model.save_model('1.bst.tmp')

    for i in range(100):
        start_timer = timer()
        df_train_X, df_train_Y = resampleTrainData(df_train)  # сделали подвыборку
        if model is None:
            logger.debug('First model.fit()')
            model = xgboost.XGBClassifier(max_depth=max_depth, n_estimators=n_estimators, subsample=subsample,
                                          colsample_bylevel=0.98,
                                          colsample_bytree=0.98,
                                          learning_rate=0.4,
                                          min_child_weight=9.0,
                                          gamma=3)
            # model = xgboost.XGBRegressor(max_depth=max_depth, n_estimators=n_estimators, subsample=subsample)
            model.fit(df_train_X, df_train_Y, verbose=True)
            model.save_model('1.bst.tmp')
        else:
            logger.debug('model.fit()')
            model.fit(df_train_X, df_train_Y, verbose=True, xgb_model='1.bst.tmp')
            model.save_model('1.bst.tmp')
        f1score = test(model, df_test.compute(), df_test_Y.compute(), save=True)
        start_timer = timer(start_timer)

    # input("Press Enter to continue...")

    print(model)

    # получение весов обученной модели (какие параметры значимы)
    import matplotlib.pyplot as plt

    for a, b in sorted(zip(model.feature_importances_, df_train_X.columns)):
        print(a, b, sep='\t\t')
    plot_importance(model, max_num_features=25)
    plt.show()
