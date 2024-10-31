from sklearn.ensemble import RandomForestClassifier
from src.pychirps.pandas_utils.data_encoding import PandasEncoder
from pychirps.extract_paths.forest_explorer import ForestExplorer
from src.pychirps.extract_paths.classification_trees import random_forest_paths_factory
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
    encoder = PandasEncoder(
        dp.cervicalb_pd.features.iloc[:600,], dp.cervicalb_pd.target.iloc[:600]
    )
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


@pytest.fixture(scope="session")
def nursery_enc():
    encoder = PandasEncoder(
        dp.nursery_pd.features.iloc[:600,], dp.nursery_pd.target.iloc[:600]
    )
    encoder.fit()
    transformed_features, transformed_target = encoder.transform()
    return PreparedData(
        features=transformed_features, target=transformed_target, encoder=encoder
    )

@pytest.fixture(scope="session")
def nursery_rf(nursery_enc):
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(nursery_enc.features, nursery_enc.target)
    return model


@pytest.fixture
def cervical_rf_paths(cervicalb_enc, cervicalb_rf):
    forest_explorer = ForestExplorer(cervicalb_rf, cervicalb_enc.encoder)
    instance = dp.cervicalh_pd.features.iloc[0]
    instance32 = instance.to_numpy().astype(np.float32).reshape(1, -1)
    return random_forest_paths_factory(forest_explorer, instance32)
