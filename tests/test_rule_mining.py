import pytest
from dataclasses import asdict
from tests.fixture_helper import (
    assert_dict_matches_fixture,
    convert_native,
)
from app.pychirps.rule_mining.pattern_miner import NodePattern
import numpy as np


def test_rule_miner_init_patterns(cervicalb_rule_miner):  # noqa # mypy can't cope with pytest fixtures
    # for p, w in zip(cervicalb_rule_miner.patterns, cervicalb_rule_miner.weights):
    #     print(p, w)
    # assert False
    assert_dict_matches_fixture(
        {
            i: {
                "pattern": [convert_native(asdict(p)) for p in pattern],
            }
            for i, (pattern) in enumerate(cervicalb_rule_miner.patterns)
        },
        "rule_miner_patterns",
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


def test_pattern_equality():
    # fail if any special __eq__ method is defined incorrectly
    assert NodePattern(feature=0, threshold=22.0, leq_threshold=True) == NodePattern(
        feature=0, threshold=22.0, leq_threshold=True
    )
    assert NodePattern(feature=0, threshold=22.0, leq_threshold=True) != NodePattern(
        feature=0, threshold=22.0, leq_threshold=False
    )
    assert NodePattern(feature=0, threshold=22.0, leq_threshold=True) != NodePattern(
        feature=0, threshold=23.0, leq_threshold=True
    )
    assert NodePattern(feature=0, threshold=22.0, leq_threshold=True) != NodePattern(
        feature=1, threshold=22.0, leq_threshold=True
    )


def test_pattern_list_prune(cervicalb_rule_miner):  # noqa # mypy can't cope with pytest fixtures
    nodes = cervicalb_rule_miner.patterns[:10]
    # test pruning logic, get a known covered pattern e.g. first node of first pattern and append it for a new tuple of node pattern
    covering_pattern = (nodes[0][0],)
    # remove any instance of the covering pattern from the list of nodes
    nodes = tuple(
        node_pattern for node_pattern in nodes if node_pattern != covering_pattern
    )
    # add the covering pattern to the end of the tuple of nodes
    nodes_add_singleton = nodes + (covering_pattern,)
    assert nodes_add_singleton != nodes
    # test pruning logic with a pattern that is covered by the first node of the first pattern
    pruned_patterns = cervicalb_rule_miner.prune_covered_patterns(
        covering_pattern=covering_pattern, patterns=nodes_add_singleton
    )
    # were all the covering patterns pruned?
    assert pruned_patterns == nodes

    # test pruning logic with first pattern that is two long
    two_patterns = tuple(
        node_pattern for node_pattern in nodes if len(node_pattern) == 2
    )
    non_covered_singleton = (
        NodePattern(feature=999999, threshold=np.inf, leq_threshold=True),
    )
    pruned_patterns = nodes + (non_covered_singleton,)
    for tp in two_patterns:
        pruned_patterns = cervicalb_rule_miner.prune_covered_patterns(
            covering_pattern=tp, patterns=pruned_patterns
        )
    # were all the covering patterns pruned?
    assert not any(cp in pruned_patterns for cp in two_patterns)

    # was the non_covered_singleton not pruned?
    assert non_covered_singleton in pruned_patterns
