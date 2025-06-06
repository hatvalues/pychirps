import pytest
import numpy as np
from numpy.random import seed
import app.pychirps.rule_mining.rule_utilities as rutils
from app.pychirps.rule_mining.pattern_miner import NodePattern
from tests.fixture_helper import (
    load_yaml_fixture_file,
    assert_dict_matches_fixture,
    convert_native,
)


def test_apply_rule(cervicalb_enc):
    fixture = load_yaml_fixture_file("nodes_example_1")
    rule = tuple(rutils.NodePattern(**node) for node in fixture["nodes"])
    features_in_rule = tuple(node["feature"] for node in fixture["nodes"])

    indices = rutils.apply_rule(rule, cervicalb_enc.features[:10])
    not_indices = [i for i in range(10) if i not in indices]

    matching_features = cervicalb_enc.features[np.ix_(indices, features_in_rule)]
    not_matching_features = cervicalb_enc.features[
        np.ix_(not_indices, features_in_rule)
    ]

    assert_dict_matches_fixture(
        {
            "matching_features": matching_features.tolist(),
            "not_matching_features": not_matching_features.tolist(),
        },
        "matching_features_example_1",
    )


def test_stability_cervical(cervicalb_enc, cervicalb_rf):
    sample_features = cervicalb_enc.features[:10]
    pred_labels = cervicalb_rf.predict(sample_features)

    explain_instance = cervicalb_enc.features[10, :]
    explain_label = cervicalb_rf.predict(explain_instance.reshape(1, -1))

    stability_score = rutils.stability(
        y_pred=explain_label,
        z_pred=pred_labels,
        Z=sample_features,
        pattern=tuple(
            rutils.NodePattern(**node)
            for node in load_yaml_fixture_file("nodes_example_1")["nodes"]
        ),
        K=2,
    )
    assert stability_score == 0.75


def test_stability_nursery(nursery_enc, nursery_rf):
    sample_features = nursery_enc.features[:10]
    pred_labels = nursery_rf.predict(sample_features)

    explain_instance = nursery_enc.features[10, :]
    explain_label = nursery_rf.predict(explain_instance.reshape(1, -1))

    stability_score = rutils.stability(
        y_pred=explain_label,
        z_pred=pred_labels,
        Z=sample_features,
        pattern=tuple(
            rutils.NodePattern(**node)
            for node in load_yaml_fixture_file("nodes_example_2")["nodes"]
        ),
        K=4,
    )
    assert stability_score == 0.25


def test_true_negative_rate_cervical(cervicalb_enc, cervicalb_rf):
    sample_features = cervicalb_enc.features[1:101]
    pred_labels = cervicalb_rf.predict(sample_features)

    explain_instance = cervicalb_enc.features[0, :]
    explain_label = cervicalb_rf.predict(explain_instance.reshape(1, -1))

    tnr = rutils.true_negative_rate(
        y_pred=explain_label,
        z_pred=pred_labels,
        Z=sample_features,
        pattern=tuple(
            rutils.NodePattern(**node)
            for node in load_yaml_fixture_file("nodes_example_1")["nodes"]
        ),
    )
    assert tnr == 0.5


def test_true_negative_rate_nursery(nursery_enc, nursery_rf):
    sample_features = nursery_enc.features[1:101]
    pred_labels = nursery_rf.predict(sample_features)

    explain_instance = nursery_enc.features[0, :]
    explain_label = nursery_rf.predict(explain_instance.reshape(1, -1))

    tnr = rutils.true_negative_rate(
        y_pred=explain_label,
        z_pred=pred_labels,
        Z=sample_features,
        pattern=tuple(
            rutils.NodePattern(**node)
            for node in load_yaml_fixture_file("nodes_example_2")["nodes"]
        ),
    )
    assert tnr == 0.7307692307692307


def test_exclusive_coverage_cervical(cervicalb_enc, cervicalb_rf):
    sample_features = cervicalb_enc.features[1:101]
    pred_labels = cervicalb_rf.predict(sample_features)

    explain_instance = cervicalb_enc.features[0, :]
    explain_label = cervicalb_rf.predict(explain_instance.reshape(1, -1))

    ec = rutils.exclusive_coverage(
        y_pred=explain_label,
        z_pred=pred_labels,
        Z=sample_features,
        pattern=tuple(
            rutils.NodePattern(**node)
            for node in load_yaml_fixture_file("nodes_example_1")["nodes"]
        ),
        K=2,
    )
    assert ec == 0.35784313725490197


def test_exclusive_coverage_nursery(nursery_enc, nursery_rf):
    sample_features = nursery_enc.features[1:101]
    pred_labels = nursery_rf.predict(sample_features)

    explain_instance = nursery_enc.features[0, :]
    explain_label = nursery_rf.predict(explain_instance.reshape(1, -1))

    ec = rutils.exclusive_coverage(
        y_pred=explain_label,
        z_pred=pred_labels,
        Z=sample_features,
        pattern=tuple(
            rutils.NodePattern(**node)
            for node in load_yaml_fixture_file("nodes_example_2")["nodes"]
        ),
        K=4,
    )
    assert ec == 0.21782544378698224


def test_adjusted_cardinality_weight():
    assert rutils.adjusted_cardinality_weight(3, 0.0) == 1.0
    assert rutils.adjusted_cardinality_weight(2, 1.0) == 0.5
    assert rutils.adjusted_cardinality_weight(4, 0.5) == 7 / 8
    assert rutils.adjusted_cardinality_weight(3, 2.0) == 1 / 3
    with pytest.warns(UserWarning):
        assert rutils.adjusted_cardinality_weight(3, -1) == 4 / 3


def test_entropy():
    assert rutils.entropy(p=np.array([1.0, 0.0])) == 0.0
    assert rutils.entropy(p=np.array([0.0, 1.0])) == 0.0
    assert rutils.entropy(p=np.array([0.5, 5.0])) == 0.30463609734923813
    assert np.isnan(rutils.entropy(p=np.array([0.0])))


def test_kl_div():
    seed(11211)
    assert (
        rutils.kl_div(
            np.array([95, 5], dtype=np.uint64), np.array([100, 0], dtype=np.uint64)
        )
        == 0.38045537670753693
    )
    assert (
        rutils.kl_div(
            np.array([5, 5], dtype=np.uint64), np.array([10, 0], dtype=np.uint64)
        )
        == 4.530273436644683
    )
    assert (
        rutils.kl_div(
            np.array([10, 0], dtype=np.uint64), np.array([50, 50], dtype=np.uint64)
        )
        == 0.692374607169956
    )
    with pytest.warns(UserWarning):
        rutils.kl_div(np.array([0.9, 0.1]), np.array([1.0, 0.0]))


def test_ws_dist():
    assert (
        rutils.ws_dis(
            np.array([95, 5], dtype=np.uint64), np.array([100, 0], dtype=np.uint64)
        )
        == 0.05
    )
    assert (
        rutils.ws_dis(
            np.array([5, 5], dtype=np.uint64), np.array([10, 0], dtype=np.uint64)
        )
        == 0.5
    )
    assert (
        rutils.ws_dis(
            np.array([10, 0], dtype=np.uint64), np.array([5, 5], dtype=np.uint64)
        )
        == 0.5
    )
    with pytest.warns(UserWarning):
        rutils.ws_dis(np.array([0.9, 0.1]), np.array([1.0, 0.0]))


arr_1 = np.array([1, 2, 3, 4, 20, 21, 22, 23, 30, 31, 32, 33, 50, 60, 61, 62, 63])
arr_2 = np.array([1, 1, 1, 1, 20, 20, 20, 20, 33, 33, 33, 33, 50, 62, 62, 62, 62])
arr_3 = np.array([])
arr_4 = np.ones(20)
arr_5 = np.concatenate((arr_4, arr_4 + 1.5))
arr_6 = np.copy(arr_1)
seed(11211)
np.random.shuffle(arr_6)


@pytest.mark.parametrize(
    "input_array,centering_function,fixture_name",
    [
        (
            arr_1,
            rutils.bin_centering,
            "bin_centres_1",
        ),
        (
            arr_1,
            rutils.cluster_centering,
            "cluster_centres_1",
        ),
        (
            arr_2,
            rutils.bin_centering,
            "bin_centres_2",
        ),
        (
            arr_2,
            rutils.cluster_centering,
            "cluster_centres_2",
        ),
        (
            arr_3,
            rutils.bin_centering,
            "bin_centres_3",
        ),
        (
            arr_3,
            rutils.cluster_centering,
            "cluster_centres_3",
        ),
        (
            arr_4,
            rutils.bin_centering,
            "bin_centres_4",
        ),
        (
            arr_4,
            rutils.cluster_centering,
            "cluster_centres_4",
        ),
        (
            arr_5,
            rutils.bin_centering,
            "bin_centres_5",
        ),
        (
            arr_5,
            rutils.cluster_centering,
            "cluster_centres_5",
        ),
        (
            arr_6,
            rutils.bin_centering,
            "bin_centres_6",
        ),
        (
            arr_6,
            rutils.cluster_centering,
            "cluster_centres_6",
        ),
    ],
)
def test_centering(input_array, centering_function, fixture_name):
    assert_dict_matches_fixture(
        {
            a: {
                "original": o,
                "bin_centred": c,
            }
            for a, (o, c) in enumerate(
                zip(
                    convert_native(input_array),
                    convert_native(centering_function(input_array)),
                )
            )
        },
        fixture_name,
    )


def sum_bin_centres(bin_centres):
    sum(v["bin_centred"] for v in bin_centres.values())


def test_centering_order_dont_matter():
    arr_1_bin = load_yaml_fixture_file("bin_centres_1")
    arr_6_bin = load_yaml_fixture_file("bin_centres_6")

    assert sum_bin_centres(arr_1_bin) == sum_bin_centres(arr_6_bin)

    arr_1_clusters = load_yaml_fixture_file("cluster_centres_1")
    arr_6_clusters = load_yaml_fixture_file("cluster_centres_6")

    assert sum_bin_centres(arr_1_clusters) == sum_bin_centres(arr_6_clusters)


@pytest.mark.parametrize(
    "stability_score,excl_cov_score,cardinality,blending_weight,cardinality_regularizing_weight,expected",
    [
        (0.8, 0.2, 1, 1.0, 1.0, 0.8),
        (0.8, 0.2, 1, 0.0, 1.0, 0.2),
        (0.8, 0.2, 1, 0.5, 1.0, 0.5),
        (0.8, 0.2, 2, 0.5, 1.0, 0.0),
        (0.8, 0.2, 5, 0.5, 1.0, -0.3),
    ],
)
def test_objective_function(
    stability_score,
    excl_cov_score,
    cardinality,
    blending_weight,
    cardinality_regularizing_weight,
    expected,
):
    assert rutils.objective_function(
        stability_score=stability_score,
        excl_cov_score=excl_cov_score,
        cardinality=cardinality,
        blending_weight=blending_weight,
        cardinality_regularizing_weight=cardinality_regularizing_weight,
    ) == pytest.approx(expected)


def test_pattern_list_prune():
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

    # test pruning logic
    node_patterns = [
        (NodePattern(0, 22.0, True),),
        (NodePattern(0, 22.0, False),),
        (NodePattern(0, 23.0, True),),
        (NodePattern(1, 22.0, True),),
        (NodePattern(0, 22.0, False), NodePattern(0, 22.0, True)),
        (NodePattern(0, 23.0, True), NodePattern(0, 22.0, True)),
        (NodePattern(1, 22.0, True), NodePattern(0, 22.0, True)),
        (NodePattern(1, 22.0, True), NodePattern(2, 22.0, True)),
    ]

    # test pruning logic
    covers = [
        rutils.pattern_covers_pattern(node_pattern, (NodePattern(0, 22.0, True),))
        for node_pattern in node_patterns
    ]

    assert covers == [True, False, False, False, True, True, True, False]
