import pathlib
from typing import List

import pandas
from os import path

from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier

NUM_OF_EXPERIMENTS = 96
DATA_PATH = 'results/'

SYNTHETIC_PATH = 'data/_generated/'
DATASETS = [
    ('Income', 'gender', 'capital-gain')
]
METHODS = [
    #'CTGAN',
    'TabFairGan',
    'TabFairGanConsistent'
]

CLASSIFICATION_METHODS = [
    #(RandomForestClassifier, 'Random Forest', {'max_depth': 3}),
    #(SGDClassifier, 'SGD', {})
    (GradientBoostingClassifier, 'Boosting', {'n_estimators': 100, 'learning_rate': 1.0, 'max_depth': 1})
]

REGRESSION_METHODS = [
    (RandomForestRegressor, 'Random Forest', {'max_depth': 3})
]


def get_transformers(df: pandas.DataFrame):
    transformers = dict()
    for name, dtype in df.dtypes.iteritems():
        if dtype in (object, bool):
            transformers[name] = LabelEncoder()
            transformers[name].fit(df[name].unique())
        elif dtype in (float, int):
            transformers[name] = StandardScaler()
            transformers[name].fit(df[name].to_numpy().reshape(-1, 1))
    return transformers


def classification(df: pandas.DataFrame, target_col: str, clf=None):
    X = df.loc[:, df.columns != target_col]
    y = df[target_col]
    res = list()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    for cls, name, args in CLASSIFICATION_METHODS:
        if clf is None:
            clf = cls(**args)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        y_labels = y.unique()
        average = 'binary'
        if len(y_labels):
            average = 'macro'
        result_row = {
            'class. method': name,
            'accuracy': metrics.accuracy_score(y_test, y_pred),
            'f1': metrics.f1_score(y_test, y_pred, average=average),
            'precision': metrics.precision_score(y_test, y_pred, average=average),
            'recall': metrics.recall_score(y_test, y_pred, average=average),
        }
        res.append(result_row)
    return res, clf


def regression(df: pandas.DataFrame, target_col: str) -> List[dict]:
    X = df.loc[:, df.columns != target_col]
    y = df[target_col]
    res = list()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    for cls, name, args in REGRESSION_METHODS:
        clf = cls(**args)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        result_row = {
            'regg. method': name,
            'explained_variance': metrics.explained_variance_score(y_test, y_pred),
            'max_error': metrics.max_error(y_test, y_pred),
            'r2': metrics.r2_score(y_test, y_pred),
            'mean_squared_error': metrics.mean_squared_error(y_test, y_pred)
        }
        res.append(result_row)
    return res


def main():
    for dataset_name, classification_coll, regression_col in DATASETS:
        original_df = pandas.read_csv(path.join(SYNTHETIC_PATH, dataset_name + '/', f'{dataset_name}_original.csv'))
        column_transformers = get_transformers(original_df)
        for col, encoder in column_transformers.items():
            original_df[col] = encoder.transform(original_df[col].to_numpy().reshape(-1, 1))

        classification_results = pandas.DataFrame(
            columns=['Exp. number', 'gener. method', 'class. method', 'f1', 'precision', 'recall', 'accuracy']
        )
        regression_results = pandas.DataFrame(
            columns=[
                'Exp. number', 'gener. method', 'regg. method',
                'explained_variance', 'max_error', 'r2', 'mean_squared_error'
            ]
        )

        if classification_coll:
            real_classification_res, clf = classification(original_df, classification_coll)
            for result_row in real_classification_res:
                result_row['gener. method'] = 'Real'
                result_row['Exp. number'] = 0
                classification_results = classification_results.append(result_row, ignore_index=True)
        if regression_col:
            for result_row in regression(original_df, regression_col):
                result_row['gener. method'] = 'Real'
                result_row['Exp. number'] = 0
                regression_results = regression_results.append(result_row, ignore_index=True)

        for method in METHODS:
            for num_exp in range(NUM_OF_EXPERIMENTS):
                synthetic_df = pandas.read_csv(
                    path.join(SYNTHETIC_PATH, dataset_name + '/', f'{dataset_name}_{method}_{num_exp}.csv')
                )
                for col, encoder in column_transformers.items():
                    synthetic_df[col] = encoder.transform(synthetic_df[col].to_numpy().reshape(-1, 1))
                if classification_coll:
                    synth_class_result, _ = classification(synthetic_df, classification_coll, clf)
                    for result_row in synth_class_result:
                        result_row['gener. method'] = method
                        result_row['Exp. number'] = num_exp
                        classification_results = classification_results.append(result_row, ignore_index=True)
                if regression_col:
                    for result_row in regression(synthetic_df, regression_col):
                        result_row['gener. method'] = method
                        result_row['Exp. number'] = num_exp
                        regression_results = regression_results.append(result_row, ignore_index=True)

            result_path = path.join(DATA_PATH, dataset_name + '/')
            pathlib.Path(result_path).mkdir(parents=True, exist_ok=True)
            if len(classification_results) > 0:
                classification_results.to_csv(path.join(result_path, dataset_name + '_classification.csv'), index=False)
            if len(regression_results) > 0:
                regression_results.to_csv(path.join(result_path, dataset_name + '_regression.csv'), index=False)


if __name__ == "__main__":
    main()
