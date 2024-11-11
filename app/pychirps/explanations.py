from app.pychirps.build_rules.rule_utilities import NodePattern
from functools import cached_property


class RuleParser:
    def __init__(self, feature_names: list[str]) -> None:
        self.feature_names = feature_names

    @cached_property
    def leq_segments(self):
        return [
            {True: "<=", False: ">"}
            if feature_name.startswith("num__")
            else {True: "is not", False: "is"}
            for feature_name in self.feature_names
        ]

    def parse_leq(self, node: NodePattern) -> str:
        return self.leq_segments[node.feature][node.leq_threshold]

    @staticmethod
    def parse_cat(node: tuple[str]) -> tuple[str]:
        split_name = node[0].replace("cat__", "").split("_")
        split_attrib = split_name.pop()
        return tuple(["_".join(split_name), node[1], split_attrib])

    def parse(self, pattern: tuple[NodePattern], y_pred: int) -> str:
        num_parse = [
            (
                self.feature_names[node.feature].replace("num__", ""),
                self.parse_leq(node),
                node.threshold,
            )
            for node in pattern
        ]
        cat_parse = [
            self.parse_cat(node) if node[0].startswith("cat__") else node
            for node in num_parse
        ]
        final_parse = [" ".join(str(n) for n in node) for node in cat_parse]
        return final_parse
