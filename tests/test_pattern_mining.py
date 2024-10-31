from tests.forest_paths_helper import weighted_paths  # noqa # mypy can't cope with pytest fixtures
from pychirps.build_rules.pattern_miner import PatternMiner
from tests.fixture_helper import assert_dict_matches_fixture, convert_native
from dataclasses import asdict


def test_pattern_miner(cervical_rf_paths):  # noqa # mypy can't cope with pytest fixtures
    pattern_miner = PatternMiner(forest_path=cervical_rf_paths)
    assert len(pattern_miner.paths) == 10


def test_pattern_miner_prediction(cervical_rf_paths):  # noqa # mypy can't cope with pytest fixtures
    pattern_miner = PatternMiner(forest_path=cervical_rf_paths, prediction=0)
    assert len(pattern_miner.paths) == 10


def test_pattern_miner_alt_prediction(cervical_rf_paths):  # noqa # mypy can't cope with pytest fixtures
    pattern_miner = PatternMiner(forest_path=cervical_rf_paths, prediction=1)
    assert len(pattern_miner.paths) == 0


def test_pattern_miner_weighted_paths(weighted_paths):  # noqa # mypy can't cope with pytest fixtures
    pattern_miner = PatternMiner(forest_path=weighted_paths)
    assert len(pattern_miner.paths) == 4
    assert len(set(pattern_miner.paths)) == 2  # one path was repeated thrice


def test_fp_paths(weighted_paths):
    pattern_miner = PatternMiner(forest_path=weighted_paths, min_support=0.5)
    patterns = pattern_miner.pattern_set.patterns
    assert_dict_matches_fixture(
        {
            i: [convert_native(asdict(p)) for p in pattern]
            for i, pattern in enumerate(patterns)
        },
        "frequent_patterns",
    )
