from tests.forest_paths_helper import weighted_paths  # noqa # mypy can't cope with pytest fixtures
from pychirps.build_rules.pattern_miner import PatternMiner
from src.pychirps.build_rules.rule_miner import RuleMiner
from dataclasses import asdict
from tests.fixture_helper import assert_dict_matches_fixture, convert_native


def test_rule_miner_init_patterns(cervicalb_enc, cervical_rf_paths):  # noqa # mypy can't cope with pytest fixtures
    pattern_miner = PatternMiner(forest_path=cervical_rf_paths)
    rule_miner = RuleMiner(pattern_miner, cervicalb_enc.features, cervicalb_enc.target)
    print(rule_miner.patterns[:10])
    print(rule_miner.weights[:10])
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

def test_rule_miner_hill_climb(cervical_rf_paths, cervicalb_enc):  # noqa # mypy can't cope with pytest fixtures
    pattern_miner = PatternMiner(forest_path=cervical_rf_paths)
    rule_miner = RuleMiner(pattern_miner, cervicalb_enc.features, cervicalb_enc.target)
    rule_miner.hill_climb()
    # assert False  # TODO: implement your test here