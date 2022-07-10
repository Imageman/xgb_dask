'''
почти оптимальный скрипт по обучению XGB
на вход подается CSV, а лучше Parquiet
входящие данные предобрабатываются (переименовываются колонки и удаляются ненужные колонки)
данные перемешиваются и делятся на обучающую и тестовую выборки (выборки записываются в паркет)
выполняются несколько иттераций с автоулучшениеим.
На каждой иттерации берется только часть данных, что бы влезало в память
'''
import math
import sys
from os import path
import os
from datetime import datetime
from random import random, randint
import re

import itertools
import matplotlib.pyplot as plt
import numpy
# import treelite as treelite
import xgboost
import pandas
from sklearn.metrics import accuracy_score, roc_curve, auc, average_precision_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from xgboost import plot_importance
import dask.dataframe as dd
# import logging
import pyarrow

# УКАЗЫВАЕМ функцию, которую будем использовать. Регрессор для непрерывных величин, классификатор для классификации (в первую очередь бинарная)
xgboost_func = xgboost.XGBRegressor
#xgboost_func = xgboost.XGBClassifier

# import fastparquet
# import snappy


experiment_name = "stat_all_pyram.txt"
filename = r'stat_all_pyram.txt'
final_model_filename = './1.bst.tmp'

class_0_name="Identical"    
class_1_name="Different"    

max_datasize = 50000  # максимальное число строчек для обработки за один раз
# чем меньше колонок, тем больше строчек можно обработать

do_hyperopt = False

colsample_bylevel=0.98
colsample_bytree=0.65
eta=0.044
gamma=1.2
max_depth = 8  # 11
min_child_weight=1.0
n_estimators = 80  # 10
subsample = 0.75  # 0.99



from loguru import logger
import sys

logger.remove()
logger.add("xgb_dask.log", rotation="150 MB", backtrace=True, diagnose=True)  # Automatically rotate too big file
logger.add(sys.stdout, colorize=True, format="<green>{time:HH:mm:ss}</green> <level>{message}</level>", level='INFO')

from psutil import virtual_memory

mem = virtual_memory()
if mem.total < 16e09:
    max_datasize = round(max_datasize / (17e09 / mem.total))
    logger.warning('max_datasize={}'.format(max_datasize))

import m2cgen

def exportC(model, filename):
    code = m2cgen.export_to_c_sharp(model)
    text_file = open(filename, "w")
    n = text_file.write(code)
    text_file.close()
    return

def exportPy(model, filename):
    code = m2cgen.export_to_python(model)
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


def test(model, df_test_X, df_test_Y, name, save=True):
    global bestF1
    if not df_test_X.isnull().sum().sum() == 0:
        logger.error(f"not df_test_X.isnull().sum().sum() == 0, sum is {df_test_X.isnull().sum().sum()}")
    if not df_test_Y.isnull().sum().sum() == 0:
        logger.error(f"not df_test_Y.isnull().sum().sum() == 0, sum is {df_test_Y.isnull().sum().sum()}")

    model.get_booster().dump_model('./result/anom_x_{}.dmp'.format('tmp'))
    # print("Accuracy:", end='', flush=True)
    # make predictions for test data
    y_pred = model.predict(df_test_X)
    #predictions = [round(value) for value in y_pred]
    predictions = [1 if value >0.5 else 0 for value in y_pred]
    print(f'nan count df_test_X: {df_test_X.isnull().sum().sum()}')
    print(f'nan count df_test_Y: {df_test_Y.isnull().sum().sum()}')
    print(f'nan count y_pred: {numpy.isnan(y_pred).sum()}')
    print(f'Range of values y_pred: {numpy.ptp(y_pred)}')

    from sklearn.metrics import f1_score
    f1 = f1_score(df_test_Y, predictions)
    accuracy = accuracy_score(df_test_Y, predictions)
    map = average_precision_score(df_test_Y, predictions)

    if f1> bestF1:
        confusion_name = name
    else:
        confusion_name = f'{name} mAP={map}'
    r = confusion_matrix(df_test_Y, predictions)

    plot_confusion_matrix(r, classes=[class_0_name,class_1_name], normalize = True, title=f'{confusion_name} Confusion matrix')
    

    fpr, tpr, _ = roc_curve(df_test_Y.values, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=lw,
            label="ROC curve (area = %0.5f)" % roc_auc,
        )
    logger.info("{} AUC: {}".format(name, roc_auc))
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{name}: Receiver operating characteristic")
    plt.legend(loc="lower right")
        # plt.show()
    plt.savefig("./result/" + str(name) + " ROC - {}.png".format(map))
    plt.close()
    logger.info("{} Accuracy: {}% F1: {}% mAP: {}% mean average error: {}%".format(name, accuracy * 100.0,
                                                                                           f1 * 100, map * 100,
                                                                                            100 - map * 100))
    if (save and f1 > bestF1):
        model.save_model('./result/model_x_{}.bst'.format(f1))
        model.get_booster().dump_model('./result/model_x_{}.dmp'.format(f1))
        # exportC(model, './result/anom_x_{}.CS'.format(F1) )
        bestF1 = f1
        # exportC(model, 'xgb_class_{}.CS'.format(f1))
        try:
            exportC(model, './result/xgb_class_{}.CS'.format(f1))
            exportPy(model, './result/xgb_class_{}.py'.format(f1))
        except Exception as e:
            logger.error(f'Error export model: {e}')
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
        bigDF.to_parquet(outfilename, engine='pyarrow', compression="snappy")
        logger.info('Fin    write ' + outfilename)
    return dd.read_parquet(outfilename, engine='pyarrow')


def resampleTrainData(df_train):
    logger.debug('resampleData()')
    logger.info("resampleTrainData start...........")
    # из большой кучи df_train делаем случайную подвыборку
    # sample_ratio должен быть небольшой, что бы результат влез в память XGB
    # global df_train_Y
    # global df_train_X
    global sample_ratio
    df_train_X = df_train.sample(frac=sample_ratio).compute()
    df_train_Y = df_train_X['result']
    df_train_X.drop(columns=['result'], inplace=True)
    logger.info("resampleTrainData end")
    return df_train_X, df_train_Y


def preprocessRAWdata(filename):
    # загружаем данные, удаляем ненужное, разбиваем на обучение и тест
    global df_train, df_test
    logger.info('Load data ' + filename)
    if filename.find('.par') > 0:
        dataf = dd.read_parquet(filename, engine='pyarrow')
    else:
        dataf = dd.read_csv(filename, skiprows=[1], skipinitialspace=True)

    dataf = dataf.rename(columns=lambda x: colRenamer(x))  # меняем заголовок
    if len(dataf.describe())<80*25:
        print(dataf.describe()) # выводим на экран краткое описание таблицы    
    
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

    # train_size = round( 0.9 * dataf.shape[0].compute())

    train_percent = 15 / 100  # для теста берем маленькую часть
    logger.debug(f'Split data. train_percent={train_percent}')
    df_train, df_test = dataf.random_split([1 - train_percent, train_percent])
    return df_train, df_test


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting normalize=True.
    refence:
        http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = numpy.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.4f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'./result/{title}.png')
    plt.close()


if __name__ == '__main__':
    # execute this only when run directly, not when imported!
    # Это условие нужно, что бы в Windows работал num_workers=x

    logger.info('Start ' + experiment_name)
    start_timer = timer()
    logger.debug('Old sys.getrecursionlimit={}'.format(sys.getrecursionlimit()))
    sys.setrecursionlimit(5000)
    logger.debug('New sys.getrecursionlimit={}'.format(sys.getrecursionlimit()))
    os.makedirs('data', exist_ok = True)
    os.makedirs('result', exist_ok = True)
    
    bestF1 = 0

    df_test = None
    df_train = None

    if not path.exists('./data/test.parquet'):  # ! если есть "наш" паркет, то перескакиваем
        # !включаем это, ТОЛЬКО если хотим заново загрузить сырые данные!
        df_train, df_test = preprocessRAWdata(filename)  
    else:
        logger.warning('Use data src from ./data/test.parquet' )

    df_train = reoptimize_to_Parquiet(df_train, './data/train.parquet')  # реоптимизация и сохранение или загрузка  data
    df_test = reoptimize_to_Parquiet(df_test, './data/test.parquet')  # реоптимизация и сохранение или загрузка  data

    logger.debug('Test data process')
    df_test_Y = df_test['result']
    df_test = df_test.drop('result', axis=1)

    logger.debug('Get length df_train')
    total_data = df_train.shape[0].compute()  # len(df_train)
    sample_ratio = max_datasize / total_data
    if sample_ratio > 1:
        sample_ratio = 1

    if do_hyperopt:
        logger.info('Start HyperParams optimize')
        sample_ratio = sample_ratio*0.95
        df_train_X, df_train_Y = resampleTrainData(df_train)  # сделали подвыборку
        # model = xgboost.XGBClassifier(max_depth=9, n_estimators=150, subsample=1)
        # getHyperParams(model, df_train_X, df_train_Y)
        import xgb_hyperopt
        xgb_hyperopt.xgboost_func=xgboost_func
        xgb_hyperopt.do_process(df_train_X, df_train_Y, df_test.compute(), df_test_Y.compute())
        input("Press Enter to continue...")

    logger.info(
        'Total train data {}; test data {}.  Ratio of train part {:.3f}'.format(total_data, len(df_test), sample_ratio))

    start_timer: datetime = timer(start_timer)
    logger.info('xgboost train')
    model = None

    if path.exists(final_model_filename):
        logger.warning(f'Try load and retraining {final_model_filename}')
        model = xgboost_func(max_depth=max_depth, n_estimators=n_estimators, subsample=subsample)
        model.load_model(final_model_filename)
        f1score = test(model, df_test.compute(), df_test_Y.compute(), name=experiment_name, save=False)

    max_iter = round(1/sample_ratio) + 8 
    if sample_ratio > 0.95:
        max_iter = 8
    for i in range(max_iter):
        start_timer = timer()
        df_train_X, df_train_Y = resampleTrainData(df_train)  # сделали подвыборку
        if model is None:
            logger.debug('First model.fit()')
            '''
            Params: {
            'colsample_bylevel': 0.8432347511150223, 
            'colsample_bytree': 0.8864742601005552,
            'eta': 0.03309426695712256, 
            'gamma': 1.4332841515555286, 
            'max_depth': 8.0, 
            'min_child_weight': 10.0, 
            'n_estimators': 558.0, 
            'subsample': 0.7142347574800574}
            '''

                                          
            model = xgboost_func(max_depth=max_depth, n_estimators=n_estimators, subsample=subsample,
                                          colsample_bylevel=colsample_bylevel,
                                          colsample_bytree=colsample_bytree,
                                          learning_rate=eta,
                                          min_child_weight=min_child_weight,
                                          gamma=gamma)            

            model.fit(df_train_X, df_train_Y, verbose=True)
            model.save_model(final_model_filename)

        else:
            logger.debug('existing retraining model.fit()')
            model.fit(df_train_X, df_train_Y, verbose=True, xgb_model=final_model_filename)
            model.save_model(final_model_filename)

        f1score = test(model, df_test.compute(), df_test_Y.compute(), name=experiment_name, save=True)
        start_timer = timer(start_timer)

    # input("Press Enter to continue...")

    print(model)

    # получение весов обученной модели (какие параметры значимы)
    import matplotlib.pyplot as plt

    for a, b in sorted(zip(model.feature_importances_, df_train_X.columns)):
        print(a, b, sep='\t\t')
    plot_importance(model, max_num_features=25)
    plt.show()
