from tests.forest_paths_helper import random_forest_paths, weighted_paths # noqa # mypy can't cope with pytest fixtures
from pyfpgrowth import find_frequent_patterns
from src.pychirps.build_rules.rule_mining import RuleMiner
from tests.fixture_helper import assert_dict_matches_fixture, convert_native
from dataclasses import asdict

def test_rule_miner(random_forest_paths): # noqa # mypy can't cope with pytest fixtures
    rule_miner = RuleMiner(forest_path=random_forest_paths)
    assert len(rule_miner.paths) == 10

def test_rule_miner_prediction(random_forest_paths): # noqa # mypy can't cope with pytest fixtures
    rule_miner = RuleMiner(forest_path=random_forest_paths, prediction=0)
    assert len(rule_miner.paths) == 10

def test_rule_miner_alt_prediction(random_forest_paths): # noqa # mypy can't cope with pytest fixtures
    rule_miner = RuleMiner(forest_path=random_forest_paths, prediction=1)
    assert len(rule_miner.paths) == 0

def test_rule_miner_weighted_paths(weighted_paths): # noqa # mypy can't cope with pytest fixtures
    rule_miner = RuleMiner(forest_path=weighted_paths)
    assert len(rule_miner.paths) == 4
    assert len(set(rule_miner.paths)) == 2 # one path was repeated thrice


def test_fp_paths(weighted_paths):
    rule_miner = RuleMiner(forest_path=weighted_paths)
    patterns = find_frequent_patterns(rule_miner.paths, 2)
    assert_dict_matches_fixture({i: [convert_native(asdict(k)) for k in keys] for i, keys in enumerate(patterns.keys()) if len(keys) == 1}, "frequent_patterns")
