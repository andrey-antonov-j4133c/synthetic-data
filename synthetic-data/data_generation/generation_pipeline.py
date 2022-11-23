import pathlib
from os import path
import ray

from data_generation.data_reader import DataSet
from data_generation.dataset_definition import DatasetDefinition
from data_generation.method import CTGAN, TabFiarGAN, TabFiarGANConsistent

BASE_DIR = 'data/'
DATAPATHS = [
    #'climate-data/DailyDelhiClimateTest.csv',
    #'Books/ratings.csv',
    #'IMDB/data.tsv',
    #'movie-dataset/movies_metadata.csv',
    #'movie-dataset/ratings.csv'
]

NUM_OF_EXPERIMENTS = 24
EPOCHS = 10
NUM_SAMPLES = 1000

SYNTHETIC_PATH = 'data/_generated/'

DATASETS = [
    #DatasetDefinition(
    #    name='MovieRatings',
    #    path='data/movie-dataset/movies_metadata.csv',
    #    categorical_cols=['adult', 'original_language', 'status', 'video'],
    #    to_drop=['homepage', 'imdb_id', 'original_title', 'overview', 'poster_path', 'release_date', 'title', 'tagline']
    #),
    DatasetDefinition(
        name='Income',
        path='data/income/train.csv',
        categorical_cols=['workclass', 'educational-num', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country', 'income_>50K', ],
        to_drop=['education'],
        target_col='gender'
    )
]

METHODS = [
    #CTGAN,
    TabFiarGAN,
    TabFiarGANConsistent
]


@ray.remote
def experiment(method, d, dataset, num_samples, experiments_path, exp_num):
    m = method(d.df, dataset.categorical_cols, dataset.target_col, EPOCHS)
    samples = m.sample(num_samples)
    # Write synthetic samples
    samples.to_csv(
        path.join(experiments_path, dataset.name + f'_{m.name}_{exp_num}.csv'),
        index=False
    )


def main():
    ray.init()
    for dataset in DATASETS:
        d = DataSet(dataset.path, to_drop=dataset.to_drop)
        # Sample from original dataset and save to file
        experiments_path = path.join(SYNTHETIC_PATH, dataset.name + '/')
        #pathlib.Path(experiments_path).mkdir(parents=True, exist_ok=True)
        num_samples = NUM_SAMPLES if len(d.df) > NUM_SAMPLES else len(d.df)
        num_samples = len(d.df)
        original_samples = d.df.sample(n=num_samples)
        pathlib.Path(experiments_path).mkdir(parents=True, exist_ok=True)
        original_samples.to_csv(
            path.join(experiments_path, dataset.name + '_original.csv'),
            index=False
        )
        experiments = []
        for method in METHODS:
            for exp_num in range(NUM_OF_EXPERIMENTS):
                experiments.append(experiment.remote(method, d, dataset, num_samples, experiments_path, exp_num))
        ray.get(experiments)


if __name__ == "__main__":
    main()
