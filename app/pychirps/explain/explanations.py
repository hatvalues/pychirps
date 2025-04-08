from app.pychirps.rule_mining.rule_utilities import NodePattern
from app.pychirps.data_prep.data_provider import ColumnDescriptor
from functools import cached_property
from typing import Tuple, Optional


class RuleParser:
    rule_segments = {
        "num": {
            True: "<=",
            False: ">",
        },
        "cat": {
            True: "is not",
            False: "is",
        },
        "bool": {
            True: "No",
            False: "Yes",
        },
    }

    def __init__(
        self, feature_names_enc: list[str], feature_descriptors: dict[ColumnDescriptor]
    ) -> None:
        self.feature_names_enc = feature_names_enc
        self.feature_descriptors = feature_descriptors
        self.feature_names = [k for k in self.feature_descriptors.keys()]
        self.binary_features = [
            k for k, v in self.feature_descriptors.items() if v.otype == "bool"
        ]

    @cached_property
    def feature_types(self):
        feature_types_names = [
            (feature_name.split("__")[0], feature_name.split("__")[1])
            for feature_name in self.feature_names_enc
        ]
        return [
            "bool" if ftn[1] in self.binary_features else ftn[0]
            for ftn in feature_types_names
        ]

    @cached_property
    def leq_segments(self):
        return [self.rule_segments[feature_type] for feature_type in self.feature_types]

    def match_feature_name(self, feature_name_enc: str) -> Tuple[Optional[str], str]:
        found = next((fn for fn in self.feature_names if fn in feature_name_enc), None)
        if found:
            return found, feature_name_enc.replace(found, "", 1)
        return None, feature_name_enc

    def parse_leq(self, node: NodePattern) -> str:
        return self.leq_segments[node.feature][node.leq_threshold]

    def parse_cat(self, node: tuple[str]) -> tuple[str]:
        feature_name_enc = node[0].replace("cat__", "")
        feature_name, value_name = self.match_feature_name(
            feature_name_enc=feature_name_enc
        )
        value_name = value_name.lstrip("_")
        return tuple([feature_name, node[1], value_name])

    def parse(self, pattern: tuple[NodePattern], rounding: int = 2) -> str:
        num_parse = [
            (
                self.feature_names_enc[node.feature].replace("num__", ""),
                self.parse_leq(node),
                round(node.threshold, rounding),
            )
            for node in pattern
        ]

        binary_parse = [
            (
                f"{node[0]}:",
                node[1],
            )
            if node[0] in self.binary_features
            else node
            for node in num_parse
        ]

        cat_parse = [
            self.parse_cat(node) if node[0].startswith("cat__") else node
            for node in binary_parse
        ]

        final_parse = [" ".join(str(n) for n in node) for node in cat_parse]
        return final_parse
