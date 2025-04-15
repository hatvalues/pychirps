import pytest
from app.pychirps.explain.explanations import RuleParser
from app.pychirps.rule_mining.rule_utilities import NodePattern
from tests.fixture_helper import assert_dict_matches_fixture

def test_parse_rule_cervical(cervicalb_pd, cervicalb_enc):
    rule_parser = RuleParser(
        cervicalb_enc.encoder.preprocessor.get_feature_names_out().tolist(),
        cervicalb_pd.column_descriptors,
    )
    rule = rule_parser.parse(
        (
            NodePattern(0, 3.0, True),
            NodePattern(1, 3.5, False),
            NodePattern(9, 2.7, True),
            NodePattern(10, 22.1, False),
        ),
    )
    assert_dict_matches_fixture({"rule": rule}, "rule_cervical_1")


@pytest.mark.parametrize(
    "node_pattern,expected",
    [
        (
            (
                NodePattern(0, 0.5, True),
                NodePattern(1, 0.5, False),
                NodePattern(9, 0.5, True),
                NodePattern(10, 0.5, False),
            ),
            "rule_nursery_1",
        ),
        (
            (
                NodePattern(11, 0.5, True),
                NodePattern(16, 0.5, True),
                NodePattern(17, 0.5, False),
            ),
            "rule_nursery_2",
        ),
    ],
)
def test_parse_rule_nursery(nursery_pd, nursery_enc, node_pattern, expected):
    rule_parser = RuleParser(
        nursery_enc.encoder.preprocessor.get_feature_names_out().tolist(),
        nursery_pd.column_descriptors,
    )

    rule = rule_parser.parse(node_pattern)
    assert_dict_matches_fixture({"rule": rule}, expected)
