import pytest
from dataclasses import asdict
from tests.fixture_helper import (
    assert_dict_matches_fixture,
    convert_native,
)
from app.pychirps.rule_mining.pattern_scorer import (
    RandomForestPatternScorer,
    AdaboostPatternScorer,
)
import numpy as np


def test_rf_rule_miner_init_patterns(cervicalb_rf_pattern_scorer):  # noqa # mypy can't cope with pytest fixtures
    # for p, w in zip(cervicalb_rule_miner.patterns, cervicalb_rule_miner.weights):
    #     print(p, w)
    # assert False
    assert_dict_matches_fixture(
        {
            i: {
                "pattern": [convert_native(asdict(p)) for p in pattern],
                "weight": convert_native(weight),
            }
            for i, (pattern, weight) in enumerate(
                zip(
                    cervicalb_rf_pattern_scorer.patterns,
                    cervicalb_rf_pattern_scorer.weights,
                )
            )
        },
        "pattern_score_patterns_cervicalb_rf",
    )


def test_entropy_reg_weights(cervicalb_rf_pattern_scorer):
    assert_dict_matches_fixture(
        {
            "e_weights": convert_native(
                cervicalb_rf_pattern_scorer.entropy_regularizing_weights
            )
        },
        "entropy_weights_cervicalb",
    )


def test_custom_sorted_patterns(cervicalb_rf_pattern_scorer):
    assert_dict_matches_fixture(
        {
            i: {
                "cs_pattern": [convert_native(asdict(p)) for p in pattern],
            }
            for i, (pattern) in enumerate(
                cervicalb_rf_pattern_scorer.custom_sorted_patterns
            )
        },
        "custom_sorted_patterns_cervicalb",
    )


def test_pattern_importance_score(cervicalb_rf_pattern_scorer):
    cervicalb_rf_pattern_scorer.cardinality_regularizing_weight = 1.0
    assert cervicalb_rf_pattern_scorer.pattern_importance_score(2) == 0.5
    assert (
        cervicalb_rf_pattern_scorer.pattern_importance_score(
            2, support_regularizing_weight=0.5
        )
        == 0.25
    )
    assert (
        cervicalb_rf_pattern_scorer.pattern_importance_score(
            2, entropy_regularizing_weight=0.5, support_regularizing_weight=0.5
        )
        == 0.125
    )
    cervicalb_rf_pattern_scorer.cardinality_regularizing_weight = 0.0
    assert cervicalb_rf_pattern_scorer.pattern_importance_score(2) == 1.0


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
def test_rule_miner_weights(pattern_miner_weights, expected_scaled_weights):
    pattern_scorer = RandomForestPatternScorer(
        patterns=tuple(),
        weights=pattern_miner_weights,
        y_pred=0.0,
        features=np.array([]),
        preds=np.array([]),
        classes=np.array([0, 1], dtype=np.uint8),
    )
    pattern_scorer._weights = pattern_miner_weights
    assert np.array_equal(pattern_scorer.weights, expected_scaled_weights)


@pytest.mark.parametrize(
    "max_depth",
    [
        1,
        2,
        5,
    ],
)
def test_ada_rule_miner_init_patterns(cervicalb_ada_pattern_scorer_factory, max_depth):  # noqa # mypy can't cope with pytest fixtures
    pattern_scorer = cervicalb_ada_pattern_scorer_factory(
        n_estimators=10,
        max_depth=max_depth,
    )
    assert_dict_matches_fixture(
        {
            i: {
                "pattern": [convert_native(asdict(p)) for p in pattern],
                "weight": convert_native(weight),
            }
            for i, (pattern, weight) in enumerate(
                zip(
                    pattern_scorer.patterns,
                    pattern_scorer.weights,
                )
            )
        },
        f"pattern_score_patterns_cervicalb_ada_{max_depth}",
    )


# def test_entropy_reg_weights(cervicalb_rf_pattern_scorer):
#     assert_dict_matches_fixture(
#         {
#             "e_weights": convert_native(
#                 cervicalb_rf_pattern_scorer.entropy_regularizing_weights
#             )
#         },
#         "entropy_weights_cervicalb",
#     )


# def test_custom_sorted_patterns(cervicalb_rf_pattern_scorer):
#     assert_dict_matches_fixture(
#         {
#             i: {
#                 "cs_pattern": [convert_native(asdict(p)) for p in pattern],
#             }
#             for i, (pattern) in enumerate(
#                 cervicalb_rf_pattern_scorer.custom_sorted_patterns
#             )
#         },
#         "custom_sorted_patterns_cervicalb",
#     )


# def test_pattern_importance_score(cervicalb_rf_pattern_scorer):
#     cervicalb_rf_pattern_scorer.cardinality_regularizing_weight = 1.0
#     assert cervicalb_rf_pattern_scorer.pattern_importance_score(2) == 0.5
#     assert (
#         cervicalb_rf_pattern_scorer.pattern_importance_score(
#             2, support_regularizing_weight=0.5
#         )
#         == 0.25
#     )
#     assert (
#         cervicalb_rf_pattern_scorer.pattern_importance_score(
#             2, entropy_regularizing_weight=0.5, support_regularizing_weight=0.5
#         )
#         == 0.125
#     )
#     cervicalb_rf_pattern_scorer.cardinality_regularizing_weight = 0.0
#     assert cervicalb_rf_pattern_scorer.pattern_importance_score(2) == 1.0
