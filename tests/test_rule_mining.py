from tests.forest_paths_helper import random_forest_paths, weighted_paths # noqa # mypy can't cope with pytest fixtures
from pyfpgrowth import find_frequent_patterns, generate_association_rules
from src.pychirps.build_rules.rule_mining import RuleMiner

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


# def test_fp_paths(random_forest_paths):
#     transactions = [[1, 2, 5],
#                 [2, 4],
#                 [2, 3],
#                 [1, 2, 4],
#                 [1, 3],
#                 [2, 3],
#                 [1, 3],
#                 [1, 2, 3, 5],
#                 [1, 2, 3]]

#     patterns = find_frequent_patterns(transactions, 2)

#     rules = generate_association_rules(patterns, 0.7)
#     print(patterns)
#     print(rules)
#     assert False