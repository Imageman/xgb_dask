# xgb_dask

## Скрипт для обучения на данных, которые не влезают в память

xgb_dask.py файл, это почти оптимальный скрипт по обучению XGB:
- на вход подается CSV, а лучше Parquiet;
- входящие данные предобрабатываются (переименовываются колонки и при необходимости удаляются ненужные колонки);
- данные перемешиваются и делятся на обучающую и тестовую выборки (выборки записываются в паркет);
- выполняются несколько иттераций с автоулучшениеим.
На каждой иттерации берется только часть данных, что бы влезало в память. На данный момент неоптимальное чтение из файла.

Второй - Подбор гиперпараметров для XGBoost с помощью hyperopt.

## xgb_dask

## Script for training on data that doesn't fit in memory

xgb_dask.py file, this is a near-optimal script for learning XGB:
- CSV, or preferably Parquiet, is fed as input;
- incoming data is preprocessed (columns are renamed and unnecessary columns are removed if necessary);
- the data are mixed and divided into training and test samples (the samples are written to Parquiet);
- several iterations with auto-improvement are performed.
At each iteration only a part of the data is taken to fit into memory. At the moment the reading from the file is suboptimal.

Second - Selection of hyperparameters for XGBoost using hyperopt.
