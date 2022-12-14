import pathlib
from os import path
import ray

from synthetic_data.data_generation.data_reader import DataSet
from synthetic_data.data_generation.dataset_definition import DatasetDefinition
from synthetic_data.data_generation.method import CTGAN, TabFiarGAN, TabFiarGANConsistent, CTGANConsistency

BASE_DIR = 'data/'
EXPERIMENTS_RANGE = range(0, 48)
EPOCHS = 10
RAY = True

SYNTHETIC_PATH = 'data/_generated/'

DATASETS = [
    #DatasetDefinition(
    #    name='Income',
    #    path='data/income/train.csv',
    #    categorical_cols=['workclass', 'educational-num', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country', 'income_>50K', ],
    #    to_drop=['education'],
    #    target_col='capital-gain'
    #),
    #DatasetDefinition(
    #    name='Income',
    #    path='data/income/train.csv',
    #    categorical_cols=['workclass', 'educational-num', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country', 'income_>50K', ],
    #    to_drop=['education'],
    #    target_col='gender'
    #),
    #DatasetDefinition(
    #    name='ClimateData',
    #    path='data/climate-data/DailyDelhiClimateTest.csv',
    #    categorical_cols=[],
    #    to_drop=['date'],
    #    target_col='meantemp'
    #),
    DatasetDefinition(
        name='SnP',
        path='data/SnP/sp500_stocks_reduced.csv',
        categorical_cols=[],
        to_drop=['Date', 'Symbol'],
        target_col='High'
    ),
    #DatasetDefinition(
    #    name='CreditRecord',
    #    path='data/CreditRecord/application_record_reduced.csv',
    #    categorical_cols=['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
    #                      'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL',
    #                      'OCCUPATION_TYPE'],
    #    to_drop=['ID', 'DAYS_BIRTH', 'FLAG_MOBIL'],
    #    target_col='AMT_INCOME_TOTAL'
    #),
    #DatasetDefinition(
    #    name='CreditRecord',
    #    path='data/CreditRecord/application_record_reduced.csv',
    #    categorical_cols=['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
    #                      'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL',
    #                      'OCCUPATION_TYPE'],
    #    to_drop=['ID', 'DAYS_BIRTH', 'FLAG_MOBIL'],
    #    target_col='FLAG_OWN_REALTY'
    #)
]

METHODS = [
    #CTGAN,
    #CTGANConsistency,
    TabFiarGAN,
    #TabFiarGANConsistent
]


if RAY:
    @ray.remote
    def experiment(method, d, dataset, num_samples, experiments_path, exp_num):
        m = method(d.df, dataset.categorical_cols, dataset.target_col, EPOCHS)
        samples = m.sample(num_samples)
        # Write synthetic samples
        samples.to_csv(
            path.join(experiments_path, dataset.name + f'_{m.name}_{exp_num}_{dataset.target_col}.csv'),
            index=False
        )
else:
    def experiment(method, d, dataset, num_samples, experiments_path, exp_num):
        m = method(d.df, dataset.categorical_cols, dataset.target_col, EPOCHS)
        samples = m.sample(num_samples)
        # Write synthetic samples
        samples.to_csv(
            path.join(experiments_path, dataset.name + f'_{m.name}_{exp_num}_{dataset.target_col}.csv'),
            index=False
        )


def main():
    if RAY:
        ray.init()
    for dataset in DATASETS:
        d = DataSet(dataset.path, to_drop=dataset.to_drop)
        # Sample from original dataset and save to file
        experiments_path = path.join(SYNTHETIC_PATH, dataset.name + '/')
        num_samples = len(d.df)
        original_samples = d.df.sample(n=num_samples)
        pathlib.Path(experiments_path).mkdir(parents=True, exist_ok=True)
        original_samples.to_csv(
            path.join(experiments_path, dataset.name + f'_{dataset.target_col}_original.csv'),
            index=False
        )
        experiments = []
        for method in METHODS:
            for exp_num in EXPERIMENTS_RANGE:
                if RAY:
                    experiments.append(experiment.remote(method, d, dataset, num_samples, experiments_path, exp_num))
                else:
                    experiment(method, d, dataset, num_samples, experiments_path, exp_num)
        if RAY:
            ray.get(experiments)


if __name__ == "__main__":
    main()
