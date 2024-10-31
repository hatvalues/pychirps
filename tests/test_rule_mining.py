from dataclasses import asdict
import numpy as np
from tests.fixture_helper import assert_dict_matches_fixture, convert_native


def test_rule_miner_init_patterns(cervicalb_rule_miner):  # noqa # mypy can't cope with pytest fixtures
    print(cervicalb_rule_miner.patterns[:10])
    print(cervicalb_rule_miner.weights[:10])
    assert all(w < 1.0 for w in cervicalb_rule_miner.weights)
    assert_dict_matches_fixture(
        {
            i: {
                "pattern": [convert_native(asdict(p)) for p in pattern],
                "weight": convert_native(weight),
            }
            for i, (pattern, weight) in enumerate(
                zip(cervicalb_rule_miner.patterns, cervicalb_rule_miner.weights)
            )
        },
        "rule_miner_patterns",
    )


def test_rule_miner_hill_climb(cervicalb_rule_miner):  # noqa # mypy can't cope with pytest fixtures
    cervicalb_rule_miner.hill_climb()
    assert False  # TODO: implement your test here
