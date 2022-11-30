from typing import List

import pandas
from os import path
import json


DTYPES = {
    bool: 'bool',
    float: 'float32',
    int: 'int32'
}
EXCLUDED_DTYPES = (dict, list)


class DataSet:
    def __init__(self, file_path: str, to_drop: List[str] = None):
        self.filepath, self.extension = path.splitext(file_path)
        self.df: pandas.DataFrame = self.read()
        self.cleanup(to_drop)
        self.cast_types()

    def read(self) -> pandas.DataFrame:
        sep = None
        if self.extension == 'tsv':
            sep = '\t'
        return pandas.read_csv(self.filepath + self.extension, sep=sep, header=0)

    def cast_types(self):
        def is_json(value):
            try:
                json.loads(value)
            except ValueError as e:
                return False
            return True

        types = dict()
        for i, row in self.df.iterrows():
            for name, item in zip(row.index, row):
                t = type(item)
                if item in ('False', 'True'):
                    t = bool
                if t == str:
                    if is_json(str.replace(item, '\'', '"')):
                        t = dict
                if t in EXCLUDED_DTYPES:
                    continue
                types[name] = DTYPES.get(t, 'object')
            break
        self.df = self.df[types.keys()]
        self.df = self.df.astype(types)

    def cleanup(self, to_drop: List[str]) -> None:
        self.df.dropna(axis=0, inplace=True)
        if to_drop:
            for col in to_drop:
                self.df.drop(col, inplace=True, axis=1)
