from tests.forest_paths_helper import weighted_paths  # noqa # mypy can't cope with pytest fixtures
from pychirps.build_rules.pattern_miner import PatternMiner
from tests.fixture_helper import assert_dict_matches_fixture, convert_native
from dataclasses import asdict


def test_pattern_miner(cervicalb_pattern_miner):  # noqa # mypy can't cope with pytest fixtures
    assert len(cervicalb_pattern_miner.paths) == 10


def test_pattern_miner_prediction(cervicalb_rf_paths, cervicalb_enc):  # noqa # mypy can't cope with pytest fixtures
    pattern_miner = PatternMiner(
        forest_path=cervicalb_rf_paths,
        feature_names=cervicalb_enc.encoder.preprocessor.get_feature_names_out().tolist(),
        prediction=0,
    )
    assert len(pattern_miner.paths) == 10


def test_pattern_miner_alt_prediction(cervicalb_rf_paths, cervicalb_enc):  # noqa # mypy can't cope with pytest fixtures
    pattern_miner = PatternMiner(
        forest_path=cervicalb_rf_paths,
        feature_names=cervicalb_enc.encoder.preprocessor.get_feature_names_out().tolist(),
        prediction=1,
    )
    assert len(pattern_miner.paths) == 0


def test_pattern_miner_weighted_paths(weighted_paths):  # noqa # mypy can't cope with pytest fixtures
    pattern_miner = PatternMiner(
        forest_path=weighted_paths,
        feature_names=["num__first", "cat__second", "num__third"],
    )
    assert len(pattern_miner.paths) == 4
    assert len(set(pattern_miner.paths)) == 2  # one path was repeated thrice


def test_fp_paths(weighted_paths):
    pattern_miner = PatternMiner(
        forest_path=weighted_paths,
        feature_names=["num__first", "cat__second", "num__third"],
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


def test_pattern_miner_discretize(cervicalb_pattern_miner):  # noqa # mypy can't cope with pytest fixtures
    discretized_centred = cervicalb_pattern_miner.descretize_continuous_thresholds()
    assert_dict_matches_fixture(
        {k.item(): convert_native(v) for k, v in discretized_centred[0].items()},
        "discretized_centred_leq",
    )
    assert_dict_matches_fixture(
        {k.item(): convert_native(v) for k, v in discretized_centred[1].items()},
        "discretized_centred_gt",
    )
