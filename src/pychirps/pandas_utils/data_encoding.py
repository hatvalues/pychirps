import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer


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
                ),  # Numerical features can pass as is
                ("cat", self.onehot_encoder, categorical_cols),
            ]
        )
        self.label_encoder = LabelEncoder()

    def fit(self) -> None:
        self.preprocessor.fit(self.features)
        self.label_encoder.fit(self.target)

    def transform(self) -> tuple[np.ndarray, np.ndarray]:
        features_encoded = self.preprocessor.transform(self.features)
        target_encoded = self.label_encoder.transform(self.target)
        return features_encoded, target_encoded

    def fit_transform(
        self, features: pd.DataFrame, target: pd.Series
    ) -> tuple[np.ndarray, np.ndarray]:
        self.features = features.copy()
        self.target = target.copy()
        self.fit()
        return self.transform()
