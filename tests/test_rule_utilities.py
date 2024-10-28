from src.pychirps.build_rules.rule_utilities import NodePattern, apply_rule, stability
from tests.fixture_helper import load_yaml_fixture_file, assert_dict_matches_fixture
import numpy as np


def test_apply_rule(cervicalb_enc):

    fixture = load_yaml_fixture_file("nodes_example_1")
    rule = tuple(NodePattern(**node) for node in fixture["nodes"])
    features_in_rule = tuple(node["feature"] for node in fixture["nodes"])

    indices = apply_rule(rule, cervicalb_enc.features[:10])
    not_indices = [i for i in range(10) if i not in indices]

    matching_features = cervicalb_enc.features[np.ix_(indices, features_in_rule)]
    not_matching_features = cervicalb_enc.features[np.ix_(not_indices, features_in_rule)]

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

    stability_score = stability(
        y_pred = explain_label,
        z_pred = pred_labels,
        Z = sample_features,
        pattern= tuple(NodePattern(**node) for node in load_yaml_fixture_file("nodes_example_1")["nodes"]),
        K = 2
    )
    assert stability_score == 0.75

def test_stability_nursery(nursery_enc, nursery_rf):
    sample_features = nursery_enc.features[:10]
    pred_labels = nursery_rf.predict(sample_features)

    explain_instance = nursery_enc.features[10, :]
    explain_label = nursery_rf.predict(explain_instance.reshape(1, -1))

    stability_score = stability(
        y_pred = explain_label,
        z_pred = pred_labels,
        Z = sample_features,
        pattern= tuple(NodePattern(**node) for node in load_yaml_fixture_file("nodes_example_2")["nodes"]),
        K = 4
    )
    assert stability_score == 0.25
