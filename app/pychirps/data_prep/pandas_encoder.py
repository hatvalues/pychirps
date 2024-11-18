import numpy as np
import pandas as pd
import polars as pl
from typing import Optional, Union, Any
from dataclasses import dataclass, field
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

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
    column_descriptors: dict[str, Any] = field(default_factory=dict)

class PandasEncoder:
    def __init__(self, features: pd.DataFrame, target: pd.Series):
        self.features = features.copy()
        self.target = target.copy()
        categorical_cols = self.features.select_dtypes(
            include=["object", "category"]
        ).columns
        numerical_cols = self.features.select_dtypes(
            include=["int64", "float64", "bool"]
        ).columns
        self.onehot_encoder = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False
        )
        self.preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    "passthrough",
                    numerical_cols,
                ),
                ("cat", self.onehot_encoder, categorical_cols),
            ]
        )
        self.label_encoder = LabelEncoder()

    def fit(self) -> None:
        self.preprocessor.fit(self.features)
        self.label_encoder.fit(self.target)

    def transform(
        self, features: Optional[np.ndarray] = None, target: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if (features is None) ^ (target is None):  # ^ is XOR
            raise ValueError("Features and target must both be given, or neither.")
        if features is None:
            return self.preprocessor.transform(
                self.features
            ), self.label_encoder.transform(self.target)
        if (len(features.shape) == 1 and len(target) == 1) or (
            features.shape[0] == len(target)
        ):
            return self.preprocessor.transform(features), self.label_encoder.transform(
                target
            )
        raise ValueError(
            "Target must be the same length as the number of rows in features"
        )

    def fit_transform(
        self, features: pd.DataFrame, target: pd.Series
    ) -> tuple[np.ndarray, np.ndarray]:
        self.features = features.copy()
        self.target = target.copy()
        self.fit()
        return self.transform()
    
def get_fitted_encoder_pd(data_provider_pd: DataProvider, n: Optional[int] = None) -> PandasEncoder:
    slc = slice(n)
    encoder = PandasEncoder(data_provider_pd.features.iloc[slc,], data_provider_pd.target.iloc[slc])
    encoder.fit()
    return encoder
