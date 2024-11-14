from app.pychirps.extract_paths.classification_trees import random_forest_paths_factory
from app.pychirps.prepare_data.pandas_encoder import PandasEncoder
from app.pychirps.extract_paths.forest_explorer import ForestExplorer
from app.pychirps.rule_mining.pattern_miner import PatternMiner
from app.pychirps.rule_mining.rule_miner import RuleMiner
from sklearn.ensemble import RandomForestClassifier
import data_preprocs.data_providers as dp
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
def cervicalb_enc():
    encoder = PandasEncoder(
        dp.cervicalb_pd.features.iloc[:600,], dp.cervicalb_pd.target.iloc[:600]
    )
    encoder.fit()
    transformed_features, transformed_target = encoder.transform()
    unseen_instance_features, unseen_instance_target = encoder.transform(
        dp.cervicalb_pd.features.iloc[600:601,], dp.cervicalb_pd.target.iloc[600:601]
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
    model = RandomForestClassifier(n_estimators=10, random_state=DEFAULT_RANDOM_SEED)
    model.fit(cervicalb_enc.features, cervicalb_enc.target)
    return model


@pytest.fixture
def cervicalb_rf_paths(cervicalb_enc, cervicalb_rf):
    forest_explorer = ForestExplorer(cervicalb_rf, cervicalb_enc.encoder)
    instance = dp.cervicalh_pd.features.iloc[0]
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
def nursery_enc():
    encoder = PandasEncoder(
        dp.nursery_pd.features.iloc[:600,], dp.nursery_pd.target.iloc[:600]
    )
    encoder.fit()
    transformed_features, transformed_target = encoder.transform()
    unseen_instance_features, unseen_instance_target = encoder.transform(
        dp.nursery_pd.features.iloc[600:601,], dp.nursery_pd.target.iloc[600:601]
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
    model = RandomForestClassifier(n_estimators=10, random_state=DEFAULT_RANDOM_SEED)
    model.fit(nursery_enc.features, nursery_enc.target)
    return model


@pytest.fixture
def nursery_rf_paths(nursery_enc, nurseryb_rf):
    forest_explorer = ForestExplorer(nursery_rf, nursery_enc.encoder)
    instance = dp.nursery_pd.features.iloc[0]
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
