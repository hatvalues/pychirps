from src.pychirps.build_rules.rule_utilities import NodePattern
from src.pychirps.pandas_utils.data_encoding import PandasEncoder


class RuleParser:
    def __init__(self, encoder: PandasEncoder) -> None:
        self.encoder = encoder

    @staticmethod
    def parse_leq(node: NodePattern) -> str:
        if node.leq_threshold:
            return "<="
        return ">"

    def parse_feature(self, node: NodePattern) -> str:
        return self.encoder.preprocessor.inverse_transform(node.feature)

    def parse_rule(self, pattern: tuple[NodePattern], y_pred: int) -> str:
        return [
            f"{self.parse_feature(node)} {self.parse_leq(node)} {node.threshold}"
            for node in pattern
        ]
