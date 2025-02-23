import pytest
from dataclasses import asdict
from tests.fixture_helper import assert_dict_matches_fixture, convert_native
import numpy as np


def test_rule_miner_init_patterns(cervicalb_rule_miner):  # noqa # mypy can't cope with pytest fixtures
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


def test_entropy_reg_weights(cervicalb_rule_miner):
    assert_dict_matches_fixture(
        {
            "e_weights": convert_native(
                cervicalb_rule_miner.entropy_regularizing_weights
            )
        },
        "entropy_weights_cervicalb",
    )


weights1 = tuple([1.0, 2.0, 3.0, 4.0, 5.0])
weights2 = tuple([1.0, 1.0, 1.0, 1.0, 1.0])
weights3 = tuple(w / 10.0 for w in weights1)  # max weight is 0.5
weights4 = tuple(w / 5.0 for w in weights1)  # max weight is 1.0
weights5 = tuple(w / 2.0 for w in weights1)  # max weight is 2.5
weights6 = tuple(w / 10.0 for w in weights2)  # max weight is 0.1 all same
weights7 = tuple(w * 2.0 for w in weights2)  # max weight is 2.0 all same
expected1 = [0.2, 0.4, 0.6, 0.8, 1.0]
expected2 = [1.0, 1.0, 1.0, 1.0, 1.0]


@pytest.mark.parametrize(
    "pattern_miner_weights,expected_scaled_weights",
    [
        (weights1, expected1),
        (weights2, expected2),
        (weights3, expected1),
        (weights4, expected1),
        (weights5, expected1),
        (weights6, expected2),
        (weights7, expected2),
    ],
)
def test_rule_miner_weights(
    pattern_miner_weights, expected_scaled_weights, cervicalb_rule_miner
):
    cervicalb_rule_miner._weights = pattern_miner_weights
    assert np.array_equal(cervicalb_rule_miner.weights, expected_scaled_weights)


def test_custom_sorted_patterns(cervicalb_rule_miner):
    assert_dict_matches_fixture(
        {
            p: [convert_native(asdict(node)) for node in pattern]
            for p, pattern in enumerate(cervicalb_rule_miner.custom_sorted_patterns)
        },
        "custom_sorted_patterns_cervicalb",
    )


def test_rule_miner_hill_climb(cervicalb_rule_miner):  # noqa # mypy can't cope with pytest fixtures
    cervicalb_rule_miner.hill_climb()
    assert cervicalb_rule_miner.best_stability == 0.9620253164556962
    assert cervicalb_rule_miner.best_excl_cov == 0.11843238587424634
    assert_dict_matches_fixture(
        {
            p: convert_native(asdict(node))
            for p, node in enumerate(cervicalb_rule_miner.best_pattern)
        },
        "patterns_cervicalb_hill_climb",
    )

    cervicalb_rule_miner.hill_climb(blending_weight=0.0)
    assert cervicalb_rule_miner.best_stability == 0.9554140127388535
    assert cervicalb_rule_miner.best_excl_cov == 0.20025839793281655
    assert_dict_matches_fixture(
        {
            p: convert_native(asdict(node))
            for p, node in enumerate(cervicalb_rule_miner.best_pattern)
        },
        "patterns_cervicalb_hill_climb_excl_cov_weighted",
    )
