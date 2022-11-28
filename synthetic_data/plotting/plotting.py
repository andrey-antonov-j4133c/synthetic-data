import pathlib
from os import path
from typing import List

import pandas as pd

import matplotlib
import matplotlib.pyplot as plt


matplotlib.rcParams.update({'font.size': 14})

NUM_OF_EXPERIMENTS = 24
DATA_PATH = 'results/'
SAVE = False

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
        metric_name: str,
        fig_path: str
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
    plt.legend()

    plt.title(f"{metric_name} score, real vs generated.\n {efficacy_method}")
    if SAVE:
        plt.savefig(path.join(fig_path, f'time-plot {efficacy_method}{metric_name}.svg'))
        plt.clf()
    else:
        plt.show()


def box_plot(
        real_values: pd.DataFrame,
        df: pd.DataFrame,
        methods: List[str],
        efficacy_method: str,
        fig_path: str
):
    for i, metric in enumerate(CLS_METRICS):
        metric_values = df[df['gener. method'].isin(methods)][['Exp. number', 'gener. method', metric]]
        metric_values = metric_values.groupby(['Exp. number', 'gener. method'])[metric].first().unstack()
        metric_values.plot(kind='box', title=f'{efficacy_method}\n{metric}')
        real_value = real_values[metric][0]
        plt.axhline(y=real_value, label=f"Real {metric} value", color='red')
        if SAVE:
            plt.savefig(path.join(fig_path, f'box-plot -{efficacy_method}-{metric}.svg'))
            plt.clf()
        else:
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
                    num_of_experiments=NUM_OF_EXPERIMENTS,
                    fig_path=plots_path
                )
            box_plot(
                real_values=cls_real_result,
                df=exp_df,
                methods=METHODS,
                efficacy_method=cls_method,
                fig_path=plots_path
            )


if __name__ == "__main__":
    main()
