from app.pychirps.path_mining.classification_trees import random_forest_paths_factory
from app.pychirps.data_prep.pandas_encoder import get_fitted_encoder_pd, PandasEncoder
from app.pychirps.path_mining.forest_explorer import ForestExplorer
from app.pychirps.rule_mining.pattern_miner import PatternMiner
from app.pychirps.rule_mining.rule_miner import RuleMiner
from app.pychirps.model_prep.model_building import (
    fit_random_forest,
    RandomForestClassifier,
)
from data_preprocs.data_providers import (
    cervicalb_pd as cervicalb_pandas_provider,
    nursery_pd as nursery_pandas_provider,
)
import numpy as np
from dataclasses import dataclass
from app.config import DEFAULT_RANDOM_SEED
import pytest


@dataclass
class PreparedData:
    features: np.ndarray
    target: np.ndarray
    unseen_instance_features: np.ndarray
    unseen_instance_target: np.ndarray
    encoder: PandasEncoder


@pytest.fixture(scope="session")
def cervicalb_pd():
    return cervicalb_pandas_provider


@pytest.fixture(scope="session")
def cervicalb_enc(cervicalb_pd):
    num_instances = 600
    slc = slice(num_instances, num_instances + 1)
    encoder = get_fitted_encoder_pd(cervicalb_pd, n=num_instances)
    transformed_features, transformed_target = encoder.transform()
    unseen_instance_features, unseen_instance_target = encoder.transform(
        cervicalb_pd.features.iloc[slc,], cervicalb_pd.target.iloc[slc]
    )
    return PreparedData(
        features=transformed_features,
        target=transformed_target,
        unseen_instance_features=unseen_instance_features,
        unseen_instance_target=unseen_instance_target,
        encoder=encoder,
    )


@pytest.fixture(scope="session")
def cervicalb_rf(cervicalb_enc):
    return fit_random_forest(
        X=cervicalb_enc.features, y=cervicalb_enc.target, n_estimators=10
    )


@pytest.fixture
def cervicalb_rf_paths(cervicalb_pd, cervicalb_enc, cervicalb_rf):
    forest_explorer = ForestExplorer(cervicalb_rf, cervicalb_enc.encoder)
    instance = cervicalb_pd.features.iloc[0]
    instance32 = instance.to_numpy().astype(np.float32).reshape(1, -1)
    return random_forest_paths_factory(forest_explorer, instance32)


@pytest.fixture
def cervicalb_pattern_miner(cervicalb_rf_paths, cervicalb_enc):  # noqa # mypy can't cope with pytest fixtures
    return PatternMiner(
        forest_path=cervicalb_rf_paths,
        feature_names=cervicalb_enc.encoder.preprocessor.get_feature_names_out().tolist(),
    )


@pytest.fixture
def cervicalb_rule_miner(cervicalb_rf, cervicalb_enc, cervicalb_pattern_miner):  # noqa # mypy can't cope with pytest fixtures
    y_pred = cervicalb_rf.predict(cervicalb_enc.unseen_instance_features)[0]
    preds = cervicalb_rf.predict(cervicalb_enc.features)
    return RuleMiner(
        pattern_miner=cervicalb_pattern_miner,
        y_pred=y_pred,
        features=cervicalb_enc.features,
        preds=preds,
        classes=np.unique(cervicalb_enc.target),
    )


@pytest.fixture(scope="session")
def nursery_pd():
    return nursery_pandas_provider


@pytest.fixture(scope="session")
def nursery_enc(nursery_pd):
    num_instances = 600
    slc = slice(num_instances, num_instances + 1)
    encoder = get_fitted_encoder_pd(nursery_pd, n=num_instances)
    transformed_features, transformed_target = encoder.transform()
    unseen_instance_features, unseen_instance_target = encoder.transform(
        nursery_pd.features.iloc[slc,], nursery_pd.target.iloc[slc]
    )
    return PreparedData(
        features=transformed_features,
        target=transformed_target,
        unseen_instance_features=unseen_instance_features,
        unseen_instance_target=unseen_instance_target,
        encoder=encoder,
    )


@pytest.fixture(scope="session")
def nursery_rf(nursery_enc):
    return fit_random_forest(
        X=nursery_enc.features, y=nursery_enc.target, n_estimators=10
    )


@pytest.fixture
def nursery_rf_paths(nursery_pd, nursery_enc, nursery_rf):
    forest_explorer = ForestExplorer(nursery_rf, nursery_enc.encoder)
    instance = nursery_pd.features.iloc[0]
    instance32 = instance.to_numpy().astype(np.float32).reshape(1, -1)
    return random_forest_paths_factory(forest_explorer, instance32)


@pytest.fixture
def nursery_pattern_miner(nursery_rf_paths, nursery_enc):  # noqa # mypy can't cope with pytest fixtures
    return PatternMiner(
        forest_path=nursery_rf_paths,
        feature_names=nursery_enc.encoder.preprocessor.get_feature_names_out().tolist(),
    )


@pytest.fixture
def nursery_rule_miner(nursery_rf, nursery_enc, nursery_pattern_miner):  # noqa # mypy can't cope with pytest fixtures
    y_pred = nursery_rf.predict(nursery_enc.unseen_instance_features)[0]
    preds = nursery_rf.predict(nursery_enc.features)
    return RuleMiner(
        pattern_miner=nursery_pattern_miner,
        y_pred=y_pred,
        features=nursery_enc.features,
        preds=preds,
        classes=np.unique(nursery_enc.target),
    )
