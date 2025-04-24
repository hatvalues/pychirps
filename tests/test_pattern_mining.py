from tests.forest_paths_helper import weighted_paths  # noqa # mypy can't cope with pytest fixtures
from app.pychirps.rule_mining.pattern_miner import RandomForestPatternMiner
from tests.fixture_helper import assert_dict_matches_fixture, convert_native
from dataclasses import asdict
from itertools import cycle
import numpy as np


def test_pattern_miner(cervicalb_rf_pattern_miner):  # noqa # mypy can't cope with pytest fixtures
    assert len(cervicalb_rf_pattern_miner.paths) == 10


def test_pattern_miner_prediction(cervicalb_rf_paths, cervicalb_enc):  # noqa # mypy can't cope with pytest fixtures
    pattern_miner = RandomForestPatternMiner(
        forest_path=cervicalb_rf_paths,
        feature_names=cervicalb_enc.encoder.preprocessor.get_feature_names_out().tolist(),
        prediction=0,
    )
    assert len(pattern_miner.paths) == 10


def test_pattern_miner_alt_prediction(cervicalb_rf_paths, cervicalb_enc):  # noqa # mypy can't cope with pytest fixtures
    pattern_miner = RandomForestPatternMiner(
        forest_path=cervicalb_rf_paths,
        feature_names=cervicalb_enc.encoder.preprocessor.get_feature_names_out().tolist(),
        prediction=1,
    )
    assert len(pattern_miner.paths) == 0


def test_pattern_miner_weighted_paths(weighted_paths):  # noqa # mypy can't cope with pytest fixtures
    pattern_miner = RandomForestPatternMiner(
        forest_path=weighted_paths,
        feature_names=["num__first", "cat__second", "num__third"],
        prediction=0.0,
    )
    assert len(pattern_miner.paths) == 4
    assert len(set(pattern_miner.paths)) == 2  # one path was repeated thrice


def test_fp_paths(weighted_paths):
    pattern_miner = RandomForestPatternMiner(
        forest_path=weighted_paths,
        feature_names=["num__first", "cat__second", "num__third"],
        prediction=0.0,
        min_support=0.5,
    )
    patterns = pattern_miner.pattern_set.patterns
    assert_dict_matches_fixture(
        {
            i: [convert_native(asdict(p)) for p in pattern]
            for i, pattern in enumerate(patterns)
        },
        "frequent_patterns",
    )


def test_feature_value_generator(cervicalb_rf_pattern_miner):
    feature_values = {
        np.int64(0): [3.4, 4.5, 6.7, 8.9],
        np.int64(4): [1.2, 2.3, 3.4, 4.5, 5.6, 6.7],
    }
    feature_value_generator = cervicalb_rf_pattern_miner.feature_value_generator(
        feature_values
    )
    alternator = cycle([0, 4])
    current_values = {0: np.inf, 4: np.inf}
    final_return_values = {0: np.inf, 4: np.inf}
    while not (current_values[0] is None and current_values[4] is None):
        key = next(alternator)
        if current_values[0] is not None:
            final_return_values[0] = current_values[0]
        if current_values[4] is not None:
            final_return_values[4] = current_values[4]
        current_values[key] = next(feature_value_generator[key], None)

    assert final_return_values == {0: 8.9, 4: 6.7}


def test_pattern_miner_discretize(cervicalb_rf_pattern_miner):  # noqa # mypy can't cope with pytest fixtures
    discretized_centred = cervicalb_rf_pattern_miner.discretize_continuous_thresholds()
    assert_dict_matches_fixture(
        {
            feature.item(): [value.item() for value in values]
            for feature, values in discretized_centred[0].items()
        },
        "discretized_centred_leq",
    )
    assert_dict_matches_fixture(
        {
            feature.item(): [value.item() for value in values]
            for feature, values in discretized_centred[0].items()
        },
        "discretized_centred_gt",
    )

    assert_dict_matches_fixture(
        {
            p: [convert_native(asdict(node)) for node in path]
            for p, path in enumerate(cervicalb_rf_pattern_miner.discretized_paths)
        },
        "discretized_paths",
    )
