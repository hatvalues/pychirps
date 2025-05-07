from app.pychirps.path_mining.forest_explorer import ForestExplorer, TreePath
from sklearn.ensemble import RandomForestClassifier
from app.pychirps.data_prep.pandas_encoder import PandasEncoder
from data_preprocs.data_providers.cervical import cervicalb_pd
from tests.fixture_helper import assert_dict_matches_fixture, convert_native
from dataclasses import asdict
import numpy as np
import pytest


def test_random_forest_explorer():
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    encoder = PandasEncoder(cervicalb_pd.features, cervicalb_pd.target)
    encoder.fit()
    transformed_features, transformed_target = encoder.transform()
    model.fit(transformed_features, transformed_target)

    forest_explorer = ForestExplorer(model, encoder)
    assert len(forest_explorer.trees) == 10
    assert len(forest_explorer.tree_weights) == 10
    assert isinstance(forest_explorer.feature_names, list)
    assert isinstance(forest_explorer.trees, list)
    assert isinstance(forest_explorer.tree_weights, np.ndarray)
    assert np.all(tw == 1.0 for tw in forest_explorer.tree_weights)
    sparse_path = forest_explorer.trees[0].tree_.decision_path(
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


@pytest.mark.parametrize("max_depth", [1, 2, 5])
def test_adaboost_explorer_weights(cervicalb_ada_factory, max_depth):
    model = cervicalb_ada_factory(n_estimators=10, max_depth=max_depth)
    assert model.estimator_weights_ is not None
    assert_dict_matches_fixture(
        {
            "estimator_weights_": convert_native(model.estimator_weights_),
        },
        f"adaboost_estimator_weights_{max_depth}",
    )


@pytest.mark.parametrize(
    "max_depth,expected_sparse_path",
    [
        (
            1,
            [
                0,
                1,
            ],
        ),
        (
            2,
            [
                0,
                1,
                2,
            ],
        ),
        (
            5,
            [
                0,
                1,
                2,
                3,
                4,
                5,
            ],
        ),
    ],
)
def test_adaboost_explorer(
    cervicalb_enc, cervicalb_ada_factory, max_depth, expected_sparse_path
):
    model = cervicalb_ada_factory(n_estimators=10, max_depth=max_depth)
    assert model.estimator_weights_ is not None
    forest_explorer = ForestExplorer(model, cervicalb_enc.encoder)
    assert len(forest_explorer.trees) == 10
    assert len(forest_explorer.tree_weights) == 10
    assert isinstance(forest_explorer.feature_names, list)
    assert isinstance(forest_explorer.trees, list)
    assert isinstance(forest_explorer.tree_weights, np.ndarray)
    assert np.all(tw < 1.0 for tw in forest_explorer.tree_weights)
    assert np.all(tw > 0.0 for tw in forest_explorer.tree_weights)
    assert np.all(
        tw == mw
        for tw, mw in zip(forest_explorer.tree_weights, model.estimator_weights_)
    )
    sparse_path = forest_explorer.trees[0].tree_.decision_path(
        cervicalb_pd.features.loc[0].values.reshape(1, -1).astype(np.float32)
    )
    assert sparse_path.indices.tolist() == expected_sparse_path


def test_rf_paths(cervicalb_rf_paths):
    paths = cervicalb_rf_paths.get_paths_for_prediction(prediction=0)
    assert len(paths) == 10
    paths = cervicalb_rf_paths.get_paths_for_prediction(prediction=1)
    assert len(paths) == 0
    paths = cervicalb_rf_paths.get_paths_for_prediction(prediction=0.0)
    assert len(paths) == 10
    assert all(p[1] == 1.0 for p in paths)


def test_ada_paths(cervicalb_ada_paths_factory):
    paths = cervicalb_ada_paths_factory(max_depth=1).get_paths_for_prediction(
        prediction=0
    )
    assert len(paths) == 10
    paths = cervicalb_ada_paths_factory(max_depth=2).get_paths_for_prediction(
        prediction=0
    )
    assert len(paths) == 10
    assert all(p[1] < 1.0 for p in paths) and all(p[1] > 0.0 for p in paths)
