import pathlib
from os import path

import pandas
from pandas_profiling import ProfileReport
from random import randint


DATA_PATH = 'data/_generated/'
RESULTS_PATH = 'results/'
DATASETS = ['Income']
METHODS = [
    'CTGAN',
    'CTGANConsistency'
]
EXP_NUM = 24


def dataset_profile(name: str, method_name: str):
    result_path = path.join(RESULTS_PATH, name + '/')
    reports_path = path.join(result_path + '/reports')
    pathlib.Path(reports_path).mkdir(parents=True, exist_ok=True)

    real_df = pandas.read_csv(path.join(DATA_PATH, name, f'{name}_original.csv'))
    real_profile = ProfileReport(real_df, title=f"{name}-original")
    real_profile.to_file(path.join(reports_path, f"{name}-original"))

    exp_num = randint(0, EXP_NUM)
    fake_df = pandas.read_csv(path.join(DATA_PATH, name, f'{name}_{method_name}_{exp_num}.csv'))
    fake_profile = ProfileReport(fake_df, title=f"{name}_{method_name}")
    fake_profile.to_file(path.join(reports_path, f"{name}-{method_name}"))


if __name__ == '__main__':
    for dataset in DATASETS:
        for method in METHODS:
            dataset_profile(dataset, method)
