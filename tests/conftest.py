from sklearn.ensemble import RandomForestClassifier
from src.pychirps.pandas_utils.data_encoding import PandasEncoder
import data_preprocs.data_providers as dp
from dataclasses import dataclass
import numpy as np
import pytest


@dataclass
class PreparedData:
    features: np.ndarray
    target: np.ndarray
    encoder: PandasEncoder


@pytest.fixture(scope="session")
def cervicalb_enc():
    encoder = PandasEncoder(dp.cervicalb_pd.features, dp.cervicalb_pd.target)
    encoder.fit()
    transformed_features, transformed_target = encoder.transform()
    return PreparedData(
        features=transformed_features, target=transformed_target, encoder=encoder
    )


@pytest.fixture(scope="session")
def cervicalb_rf(cervicalb_enc):
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(cervicalb_enc.features, cervicalb_enc.target)
    return model