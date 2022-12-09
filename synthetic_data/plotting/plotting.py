import pathlib
from os import path
from typing import List

import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

from scipy.stats import ttest_ind


matplotlib.rcParams.update({'font.size': 14})

NUM_OF_EXPERIMENTS = 48
DATA_PATH = 'results/'
SAVE = True

DATASETS = [
    ('Income', 'gender', 'capital-gain'),
    ('ClimateData', None, 'meantemp'),
    ('SnP', None, 'High'),
    ('CreditRecord', 'FLAG_OWN_REALTY', 'AMT_INCOME_TOTAL')
]
METHODS = [
    #'CTGAN',
    #'CTGANConsistency',
    'TabFairGan',
    'TabFairGanConsistent'
]

CLS_METHODS = ['Boosting classifier']
REG_METHODS = ['Boosting regressor']
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
        method_names = f'{methods[0]}-{methods[1]}'
        plt.savefig(path.join(fig_path, f'{method_names}-time-plot {efficacy_method}{metric_name}.svg'))
        plt.clf()
    else:
        plt.show()


def box_plot(
        real_values: pd.DataFrame,
        df: pd.DataFrame,
        methods: List[str],
        metrics: List[str],
        efficacy_method: str,
        fig_path: str
):
    for i, metric in enumerate(metrics):
        metric_values = df[df['gener. method'].isin(methods)][['Exp. number', 'gener. method', metric]]
        metric_values = metric_values.groupby(['Exp. number', 'gener. method'])[metric].first().unstack()
        p_value = ttest_ind(
            metric_values.iloc[:, 0].to_numpy().flatten(),
            metric_values.iloc[:, 1].to_numpy().flatten()
        ).pvalue
        metric_values.plot(
            kind='box',
            title=f'{efficacy_method}\n{metric} (p_value={round(p_value, 3)})'
        )
        real_value = real_values[metric][0]
        plt.axhline(y=real_value, label=f"Real {metric} value", color='red')
        if SAVE:
            method_names = f'{methods[0]}-{methods[1]}'
            plt.savefig(path.join(fig_path, f'{method_names}-box-plot-{efficacy_method}-{metric}.svg'))
            plt.clf()
        else:
            plt.show()


def main():
    for dataset_name, cls_column, reg_column in DATASETS:
        result_path = path.join(DATA_PATH, dataset_name + '/')
        plots_path = path.join(result_path + '/plots')
        pathlib.Path(plots_path).mkdir(parents=True, exist_ok=True)

        if cls_column:
            cls_df = pd.read_csv(path.join(result_path, dataset_name + '_classification.csv'))
            cls_real_result = cls_df[cls_df['gener. method'] == 'Real']
            for cls_method in CLS_METHODS:
                exp_df = cls_df[cls_df['class. method'] == cls_method]
                for metric in CLS_METRICS:
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
                    metrics=CLS_METRICS,
                    efficacy_method=cls_method,
                    fig_path=plots_path
                )
        if reg_column:
            reg_df = pd.read_csv(path.join(result_path, dataset_name + '_regression.csv'))
            reg_real_result = reg_df[reg_df['gener. method'] == 'Real']
            for reg_method in REG_METHODS:
                exp_df = reg_df[reg_df['regg. method'] == reg_method]
                for metric in REG_METRICS:
                    time_plot(
                        real_value=reg_real_result[metric],
                        df=exp_df,
                        methods=METHODS,
                        metric_name=metric,
                        efficacy_method=reg_method,
                        num_of_experiments=NUM_OF_EXPERIMENTS,
                        fig_path=plots_path
                    )
                box_plot(
                    real_values=reg_real_result,
                    df=exp_df,
                    methods=METHODS,
                    metrics=REG_METRICS,
                    efficacy_method=reg_method,
                    fig_path=plots_path
                )


if __name__ == "__main__":
    main()
