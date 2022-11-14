from dataclasses import dataclass
from typing import List


@dataclass
class DatasetDefinition:
    name: str
    path: str
    categorical_cols: List[str]
    to_drop: List[str]
    target_col: str = None
