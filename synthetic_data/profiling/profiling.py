import pathlib
from os import path

import pandas
from pandas_profiling import ProfileReport
from random import randint


DATA_PATH = 'data/_generated/'
RESULTS_PATH = 'results/'
DATASETS = [
    ('Income', 'gender', 'capital-gain'),
    ('ClimateData', None, 'meantemp'),
    ('SnP', None, 'High'),
    ('CreditRecord', 'FLAG_OWN_REALTY', 'AMT_INCOME_TOTAL')
]
METHODS = [
    'CTGAN',
    'CTGANConsistency'
]
EXP_NUM = 5


def dataset_profile(name: str, method_name: str, col_name: str):
    result_path = path.join(RESULTS_PATH, name + '/')
    reports_path = path.join(result_path + '/reports')
    pathlib.Path(reports_path).mkdir(parents=True, exist_ok=True)

    real_df = pandas.read_csv(path.join(DATA_PATH, name, f'{name}_{col_name}_original.csv'))
    real_profile = ProfileReport(real_df, title=f"{name}-{col_name}-original")
    real_profile.to_file(path.join(reports_path, f"{name}-{col_name}-original"))

    exp_num = randint(0, EXP_NUM)
    fake_df = pandas.read_csv(path.join(DATA_PATH, name, f'{name}_{method_name}_{exp_num}_{col_name}.csv'))
    fake_profile = ProfileReport(fake_df, title=f"{name}_{method_name}_{col_name}")
    fake_profile.to_file(path.join(reports_path, f"{name}-{method_name}_{col_name}"))


if __name__ == '__main__':
    for dataset, cls_col, reg_col in DATASETS:
        for method in METHODS:
            if cls_col:
                dataset_profile(dataset, method, cls_col)
            if reg_col:
                dataset_profile(dataset, method, reg_col)
