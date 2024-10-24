from tests.forest_paths_helper import rf_paths, weighted_paths  # noqa # mypy can't cope with pytest fixtures
from pychirps.build_rules.pattern_miner import PatternMiner
from src.pychirps.build_rules.rule_miner import RuleMiner
from dataclasses import asdict
from tests.fixture_helper import assert_dict_matches_fixture, convert_native


def test_pattern_miner_weighted_paths(cervicalb_enc, rf_paths):  # noqa # mypy can't cope with pytest fixtures
    pattern_miner = PatternMiner(forest_path=rf_paths)
    rule_miner = RuleMiner(pattern_miner, cervicalb_enc.features, cervicalb_enc.target)
    assert_dict_matches_fixture({
        i: {"pattern": [convert_native(asdict(p)) for p in pattern], "weight": weight}
        for i, (pattern, weight) in enumerate(zip(rule_miner.patterns, rule_miner.weights))
    },
        "rule_miner_patterns",
    )
