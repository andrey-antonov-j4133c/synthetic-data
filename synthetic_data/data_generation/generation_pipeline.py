import pathlib
from os import path
import ray

from synthetic_data.data_generation.data_reader import DataSet
from synthetic_data.data_generation.dataset_definition import DatasetDefinition
from synthetic_data.data_generation.method import CTGAN, TabFiarGAN, TabFiarGANConsistent, CTGANConsistency

BASE_DIR = 'data/'
EXPERIMENTS_RANGE = range(0, 2)
EPOCHS = 2
RAY = True

SYNTHETIC_PATH = 'data/_generated/'

DATASETS = [
    DatasetDefinition(
        name='Income',
        path='data/income/train.csv',
        categorical_cols=['workclass', 'educational-num', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country', 'income_>50K', ],
        to_drop=['education'],
        target_col='capital-gain'
    ),
    DatasetDefinition(
        name='Income',
        path='data/income/train.csv',
        categorical_cols=['workclass', 'educational-num', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country', 'income_>50K', ],
        to_drop=['education'],
        target_col='gender'
    )
]

METHODS = [
    CTGAN,
    CTGANConsistency,
    #TabFiarGAN,
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
        #pathlib.Path(experiments_path).mkdir(parents=True, exist_ok=True)
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
