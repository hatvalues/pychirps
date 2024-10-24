from src.pychirps.extract_paths.classification_trees import (
    random_forest_paths_factory,
    ForestPath,
    TreePath,
    TreeNode,
)
from pychirps.extract_paths.forest_explorer import ForestExplorer
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


@pytest.fixture(scope="module")
def cervicalb_enc():
    encoder = PandasEncoder(dp.cervicalb_pd.features, dp.cervicalb_pd.target)
    encoder.fit()
    transformed_features, transformed_target = encoder.transform()
    return PreparedData(
        features=transformed_features, target=transformed_target, encoder=encoder
    )


@pytest.fixture(scope="module")
def cervicalb_rf(cervicalb_enc):
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(cervicalb_enc.features, cervicalb_enc.target)
    return model


@pytest.fixture
def rf_paths(cervicalb_enc, cervicalb_rf):
    forest_explorer = ForestExplorer(cervicalb_rf, cervicalb_enc.encoder)
    instance = dp.cervicalh_pd.features.iloc[0]
    instance32 = instance.to_numpy().astype(np.float32).reshape(1, -1)
    return random_forest_paths_factory(forest_explorer, instance32)


@pytest.fixture
def weighted_paths():
    tree_node_1 = TreeNode(
        feature=np.int64(27),
        feature_name="num__Dx:CIN",
        value=np.float32(0.0),
        threshold=np.float64(0.5),
        leq_threshold=np.True_,
    )
    tree_node_2 = TreeNode(
        feature=np.int64(11),
        feature_name="num__STDs",
        value=np.float32(0.0),
        threshold=np.float64(0.5),
        leq_threshold=np.True_,
    )
    tree_node_3 = TreeNode(
        feature=np.int64(1),
        feature_name="num__Number of sexual partners",
        value=np.float32(4.0),
        threshold=np.float64(1.5),
        leq_threshold=np.False_,
    )
    tree_path_1 = TreePath(
        prediction=0, nodes=(tree_node_1, tree_node_2, tree_node_3), weight=1.0
    )
    tree_path_2 = TreePath(
        prediction=1, nodes=(tree_node_2, tree_node_3, tree_node_1), weight=2.0
    )
    tree_path_3 = TreePath(
        prediction=0, nodes=(tree_node_3, tree_node_1, tree_node_2), weight=3.0
    )
    return ForestPath(prediction=0, paths=(tree_path_1, tree_path_2, tree_path_3))
