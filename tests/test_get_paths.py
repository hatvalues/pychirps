from src.pychirps.extract_paths.classification_trees import instance_tree_factory
from sklearn.ensemble import RandomForestClassifier
from src.pychirps.pandas_utils.data_encoding import PandasEncoder
import data_preprocs.data_providers as dp
import numpy as np
from dataclasses import asdict
from tests.fixture_helper import assert_dict_matches_fixture, convert_native
from tests.forest_paths_helper import rf_paths, weighted_paths  # noqa # mypy can't cope with pytest fixtures


def test_instance_tree_factory():
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    encoder = PandasEncoder(dp.cervicalb_pd.features, dp.cervicalb_pd.target)
    encoder.fit()
    transformed_features, transformed_target = encoder.transform()
    model.fit(transformed_features, transformed_target)

    instance = dp.cervicalh_pd.features.iloc[0]
    instance32 = instance.to_numpy().astype(np.float32).reshape(1, -1)
    tree = model.estimators_[0]
    feature_names = {
        i: v
        for i, v in enumerate(encoder.preprocessor.get_feature_names_out().tolist())
    }
    tree_path = instance_tree_factory(
        tree=tree, feature_names=feature_names, instance=instance32
    )
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


def test_rf_paths_factory(rf_paths):  # noqa # mypy can't cope with pytest fixtures
    assert rf_paths.prediction == 0.0
    assert rf_paths.paths[0].prediction == 0.0
    paths_by_prediction_0 = rf_paths.get_for_prediction(0)
    assert len(paths_by_prediction_0) == 10
    assert_dict_matches_fixture(
        convert_native(asdict(rf_paths.paths[0].nodes[0])),
        "basic_tree_path_0",
    )
    assert_dict_matches_fixture(
        convert_native(asdict(rf_paths.paths[0].nodes[1])),
        "basic_tree_path_1",
    )
    assert_dict_matches_fixture(
        convert_native(asdict(rf_paths.paths[0].nodes[-1])),
        "basic_tree_path_last",
    )


def test_get_weighted_paths(weighted_paths):  # noqa # mypy can't cope with pytest fixtures
    assert weighted_paths.prediction == 0
    paths_by_prediction_0 = weighted_paths.get_for_prediction(0)
    assert len(paths_by_prediction_0) == 2
    paths_by_prediction_1 = weighted_paths.get_for_prediction(1)
    assert len(paths_by_prediction_1) == 1
    assert_dict_matches_fixture(
        convert_native(asdict(weighted_paths.paths[0].nodes[0])), "weighted_tree_path_0"
    )
    assert_dict_matches_fixture(
        convert_native(asdict(weighted_paths.paths[1].nodes[0])), "weighted_tree_path_1"
    )
