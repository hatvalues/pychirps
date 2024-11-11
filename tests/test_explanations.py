from app.pychirps.explanations import RuleParser
from app.pychirps.build_rules.rule_utilities import NodePattern
from tests.fixture_helper import assert_dict_matches_fixture


def test_cat_parse(nursery_enc):
    rule_parser = RuleParser(
        nursery_enc.encoder.preprocessor.get_feature_names_out().tolist()
    )
    assert rule_parser.parse_cat(("cat__field_value", "is", 0.5)) == (
        "field",
        "is",
        "value",
    )
    assert rule_parser.parse_cat(("field_name_value", "is", 0.5)) == (
        "field_name",
        "is",
        "value",
    )


def test_parse_rule_cervical(cervicalb_enc):
    rule_parser = RuleParser(
        cervicalb_enc.encoder.preprocessor.get_feature_names_out().tolist()
    )
    rule = rule_parser.parse(
        (
            NodePattern(0, 3.0, True),
            NodePattern(1, 3.5, False),
            NodePattern(9, 2.7, True),
            NodePattern(10, 22.1, False),
        ),
        y_pred=1,
    )
    assert_dict_matches_fixture({"rule": rule}, "rule_cervical_1")


def test_parse_rule_nursery(nursery_enc):
    rule_parser = RuleParser(
        nursery_enc.encoder.preprocessor.get_feature_names_out().tolist()
    )

    rule = rule_parser.parse(
        (
            NodePattern(0, 0.5, True),
            NodePattern(1, 0.5, False),
            NodePattern(9, 0.5, True),
            NodePattern(10, 0.5, False),
        ),
        y_pred=3,
    )
    assert_dict_matches_fixture({"rule": rule}, "rule_nursery_1")
