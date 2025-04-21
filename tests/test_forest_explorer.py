from app.pychirps.path_mining.forest_explorer import ForestExplorer, TreePath
from sklearn.ensemble import RandomForestClassifier
from app.pychirps.data_prep.pandas_encoder import PandasEncoder
from data_preprocs.data_providers.cervical import cervicalb_pd
from tests.fixture_helper import assert_dict_matches_fixture, convert_native
from dataclasses import asdict
import numpy as np


def test_forest_explorer():
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    encoder = PandasEncoder(cervicalb_pd.features, cervicalb_pd.target)
    encoder.fit()
    transformed_features, transformed_target = encoder.transform()
    model.fit(transformed_features, transformed_target)

    explorer = ForestExplorer(model, encoder)
    assert len(explorer.trees) == 10
    assert len(explorer.tree_weights) == 10
    assert isinstance(explorer.feature_names, list)
    assert isinstance(explorer.trees, list)
    assert isinstance(explorer.tree_weights, np.ndarray)
    assert explorer.tree_weights.all() == 1.0
    sparse_path = explorer.trees[0].tree_.decision_path(
        cervicalb_pd.features.loc[0].values.reshape(1, -1).astype(np.float32)
    )
    assert sparse_path.indices.tolist() == [
        0,
        1,
        2,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        64,
        65,
        75,
    ]


def test_parse_tree_for_instance(cervicalb_enc, cervicalb_rf):
    forest_explorer = ForestExplorer(cervicalb_rf, cervicalb_enc.encoder)
    instance = cervicalb_enc.features[0, :]
    instance32 = instance.astype(np.float32).reshape(1, -1)

    tree_path = forest_explorer.parse_tree_for_instance(
        tree=cervicalb_rf.estimators_[0],
        instance=instance32,
        path_weight=1.0,
    )

    assert type(tree_path) == TreePath
    assert tree_path.prediction == 0.0
    assert_dict_matches_fixture(
        convert_native(asdict(tree_path.nodes[0])), "basic_tree_path_0"
    )
    assert_dict_matches_fixture(
        convert_native(asdict(tree_path.nodes[1])), "basic_tree_path_1"
    )
    assert_dict_matches_fixture(
        convert_native(asdict(tree_path.nodes[-1])), "basic_tree_path_last"
    )
