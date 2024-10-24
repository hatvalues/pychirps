from tests.forest_paths_helper import random_forest_paths, weighted_paths  # noqa # mypy can't cope with pytest fixtures
from pychirps.build_rules.pattern_miner import PatternMiner
from tests.fixture_helper import assert_dict_matches_fixture, convert_native
from dataclasses import asdict


def test_rule_miner(random_forest_paths):  # noqa # mypy can't cope with pytest fixtures
    rule_miner = PatternMiner(forest_path=random_forest_paths)
    assert len(rule_miner.paths) == 10


def test_rule_miner_prediction(random_forest_paths):  # noqa # mypy can't cope with pytest fixtures
    rule_miner = PatternMiner(forest_path=random_forest_paths, prediction=0)
    assert len(rule_miner.paths) == 10


def test_rule_miner_alt_prediction(random_forest_paths):  # noqa # mypy can't cope with pytest fixtures
    rule_miner = PatternMiner(forest_path=random_forest_paths, prediction=1)
    assert len(rule_miner.paths) == 0


def test_rule_miner_weighted_paths(weighted_paths):  # noqa # mypy can't cope with pytest fixtures
    rule_miner = PatternMiner(forest_path=weighted_paths)
    assert len(rule_miner.paths) == 4
    assert len(set(rule_miner.paths)) == 2  # one path was repeated thrice


def test_fp_paths(weighted_paths):
    print(weighted_paths)
    rule_miner = PatternMiner(forest_path=weighted_paths, min_support=0.5)
    patterns = rule_miner.pattern_set.patterns
    assert_dict_matches_fixture(
        {
            i: [convert_native(asdict(p)) for p in pattern]
            for i, pattern in enumerate(patterns)
        },
        "frequent_patterns",
    )
