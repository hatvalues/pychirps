from dataclasses import dataclass, field
from typing import Optional, Union
import pandas as pd
import polars as pl

@dataclass
class DataContainer:
    features: Union[pd.DataFrame, pl.DataFrame]
    target: Union[pd.Series, pl.Series]


@dataclass
class ColumnDescriptor:
    dtype: Optional[str] = None
    otype: Optional[str] = None
    unique_values: list = field(default_factory=list)
    min: Optional[Union[float, int]] = None
    max: Optional[Union[float, int]] = None

@dataclass
class DataProvider:
    """Base class for data providers."""

    name: str
    file_name: str
    class_col: str
    positive_class: str
    spiel: str
    sample_size: float
    features: Union[pd.DataFrame, pl.DataFrame]
    target: Union[pd.Series, pl.Series]
    column_descriptors: dict[str, ColumnDescriptor] = field(default_factory=dict)
