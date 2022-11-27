import pathlib
from os import path
from typing import List

import pandas as pd

import matplotlib.pyplot as plt


NUM_OF_EXPERIMENTS = 24
DATA_PATH = 'results/'

DATASETS = [
    ('Income', 'gender', 'capital-gain')
]
METHODS = [
    'CTGAN',
    'CTGANConsistency',
    #'TabFairGAN',
    #'TabFairGanConsistent'
]

CLS_METHODS = ['Boosting']
REG_METHODS = ['Random Forest']
CLS_METRICS = ['f1', 'precision', 'recall', 'accuracy']
REG_METRICS = ['explained_variance', 'max_error', 'r2', 'mean_squared_error']


def time_plot(
        real_value: float,
        df: pd.DataFrame,
        num_of_experiments: int,
        methods: List[str],
        efficacy_method: str,
        metric_name: str
):
    plt.plot(
        range(num_of_experiments),
        [real_value for _ in range(num_of_experiments)],
        label=f"Real {metric_name}"
    )

    for method in methods:
        method_df = df[df['gener. method'] == method]
        plt.plot(range(num_of_experiments), method_df[metric_name], label=f"{method}")

    plt.xlabel("Experiment number")
    plt.ylabel("Metric value")

    plt.title(f"{metric_name} score, real vs generated.\n {efficacy_method}")
    plt.legend()
    plt.show()


def box_plot(
        real_values: pd.DataFrame,
        df: pd.DataFrame,
        methods: List[str],
        efficacy_method: str,
):
    for method in methods:
        for i, metric in enumerate(CLS_METRICS):
            metric_values = df[df['gener. method'] == method][[metric]]
            metric_values.plot(kind='box', title=f'{method}-{efficacy_method}\n{metric}')
            real_value = real_values[metric][0]
            plt.axhline(y=real_value, label=f"Real {metric} value", color='red')
            plt.legend()
            plt.show()

def main():
    for dataset_name, cls_column, reg_column in DATASETS:
        result_path = path.join(DATA_PATH, dataset_name + '/')
        plots_path = path.join(result_path + '/plots')
        pathlib.Path(plots_path).mkdir(parents=True, exist_ok=True)
        cls_df = pd.read_csv(path.join(result_path, dataset_name + '_classification.csv'))
        reg_df = pd.read_csv(path.join(result_path, dataset_name + '_regression.csv'))

        cls_real_result = cls_df[cls_df['gener. method'] == 'Real']
        for cls_method in CLS_METHODS:
            exp_df = cls_df[cls_df['class. method'] == cls_method]
            for metric in CLS_METRICS:
                pass
                time_plot(
                    real_value=cls_real_result[metric],
                    df=exp_df,
                    methods=METHODS,
                    metric_name=metric,
                    efficacy_method=cls_method,
                    num_of_experiments=NUM_OF_EXPERIMENTS
                )
            box_plot(
                real_values=cls_real_result,
                df=exp_df,
                methods=METHODS,
                efficacy_method=cls_method,
            )


if __name__ == "__main__":
    main()
