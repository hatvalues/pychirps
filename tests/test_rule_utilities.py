from src.pychirps.pandas_utils.data_encoding import PandasEncoder
from src.pychirps.build_rules.rule_utilities import NodePattern, apply_rule
from data_preprocs.data_providers import cervicalb_pd, nursery_pd
from tests.fixture_helper import load_yaml_fixture_file, assert_dict_matches_fixture
import numpy as np

def test_apply_rule():
    encoder = PandasEncoder(cervicalb_pd.features, cervicalb_pd.target)
    encoder.fit()
    transformed_features, _ = encoder.transform()

    fixture = load_yaml_fixture_file("nodes_example_1")
    rule = tuple(NodePattern(**node) for node in fixture["nodes"])
    features_in_rule = tuple(node["feature"] for node in fixture["nodes"])

    indices = apply_rule(rule, transformed_features[:10])
    not_indices = [i for i in range(10) if i not in indices]

    matching_features = transformed_features[np.ix_(indices, features_in_rule)]
    not_matching_features = transformed_features[np.ix_(not_indices, features_in_rule)]

    assert_dict_matches_fixture(
        {"matching_features": matching_features.tolist(), "not_matching_features": not_matching_features.tolist()},
        "matching_features_example_1",
    )
