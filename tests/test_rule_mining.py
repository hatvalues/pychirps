from tests.forest_paths_helper import rf_paths, weighted_paths  # noqa # mypy can't cope with pytest fixtures
from pychirps.build_rules.pattern_miner import PatternMiner
from src.pychirps.build_rules.rule_miner import RuleMiner
from dataclasses import asdict
import numpy as np
from tests.fixture_helper import assert_dict_matches_fixture, convert_native


def test_rule_miner_init_patterns(cervicalb_enc, rf_paths):  # noqa # mypy can't cope with pytest fixtures
    pattern_miner = PatternMiner(forest_path=rf_paths)
    rule_miner = RuleMiner(pattern_miner, cervicalb_enc.features, cervicalb_enc.target)
    assert all(w < 1.0 for w in rule_miner.weights)
    assert_dict_matches_fixture(
        {
            i: {
                "pattern": [convert_native(asdict(p)) for p in pattern],
                "weight": convert_native(weight),
            }
            for i, (pattern, weight) in enumerate(
                zip(rule_miner.patterns, rule_miner.weights)
            )
        },
        "rule_miner_patterns",
    )
