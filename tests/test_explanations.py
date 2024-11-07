from src.pychirps.explanations import RuleParser
from src.pychirps.build_rules.rule_utilities import NodePattern


def test_enc(cervicalb_enc):
    print(cervicalb_enc.inverse_transform_features)
    assert False

def test_parse_rule_cervical(cervicalb_enc):
    rule_parser = RuleParser(cervicalb_enc.encoder)

    rule_parser.parse_rule(
        (
            NodePattern(0, 3.0, True),
            NodePattern(1, 3.5, False),
            NodePattern(9, 2.7, True),
            NodePattern(10, 22.1, False),
        ),
        y_pred=1
    )
