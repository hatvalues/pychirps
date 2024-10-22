from src.pychirps.extract_paths.classification_trees import ForestPath
from pyfpgrowth import find_frequent_patterns
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class NodePattern:
    feature: int
    threshold: float
    leq_threshold: bool


@dataclass
class FrequentPattern:
    antecedents: list[list[NodePattern]]
    consequent: int
    support: int


class RuleMiner:
    def __init__(self, forest_path: ForestPath):
        self.forest_path = forest_path

    # def mine_rules(self, instance: np.ndarray, min_support: int) -> list:
    #     forest_paths = get_random_forest_paths(self.forest_explorer, instance)
    #     frequent_patterns = find_frequent_patterns(
    #         [
    #             [
    #                 f"{node.feature_name} {node.leq_threshold} {node.threshold}"
    #                 for node in path.nodes
    #             ]
    #             for path in forest_paths.paths
    #         ],
    #         min_support,
    #     )
    #     return frequent_patterns
