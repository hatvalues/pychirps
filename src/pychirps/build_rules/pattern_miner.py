from src.pychirps.extract_paths.classification_trees import ForestPath
from pyfpgrowth import find_frequent_patterns
from src.pychirps.build_rules.rule_utilities import NodePattern
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Generator
import src.pychirps.build_rules.rule_utilities as rutils
import numpy as np


@dataclass(frozen=True)
class PatternSet:
    patterns: list[tuple[NodePattern]]
    weights: np.ndarray


class PatternMiner:
    def __init__(
        self,
        forest_path: ForestPath,
        feature_names: Optional[list[str]],
        prediction: Optional[int] = None,
        min_support: Optional[float] = 0.2,
    ):
        if min_support > 1:
            raise ValueError("Set min_support using a fraction")
        self.support = round(min_support * len(forest_path.paths))
        self.forest_path = forest_path
        if prediction:
            self.prediction = prediction
        else:
            self.prediction = forest_path.prediction
        self.feature_names = feature_names
        self.paths = tuple(
            tuple(
                NodePattern(
                    feature=node.feature,
                    threshold=node.threshold,
                    leq_threshold=node.leq_threshold,
                )
                for node in nodes
            )
            for nodes, weight in self.forest_path.get_for_prediction(
                prediction=self.prediction
            )
            for _ in range(int(weight))
        )

        feature_values_leq, feature_values_gt = self.discretize_continuous_thresholds()
        discretized_paths = []
        for path in self.paths:
            nodes = []
            for node in path:
                if self.feature_names[node.feature].startswith("num__") and node.leq_threshold:
                    nodes.append(NodePattern(
                        feature=node.feature,
                        threshold=next(feature_values_leq[node.feature]),
                        leq_threshold=True
                        )
                    )
                elif self.feature_names[node.feature].startswith("num__") and not node.leq_threshold:
                    nodes.append(NodePattern(
                        feature=node.feature,
                        threshold=next(feature_values_gt[node.feature]),
                        leq_threshold=False
                        )
                    )
                else:
                    nodes.append(node)
            discretized_paths.append(nodes)

        self.discretized_paths = tuple(
            tuple(
                node for node in nodes
            ) for nodes in discretized_paths
        )

        frequent_patterns = find_frequent_patterns(self.discretized_paths, self.support)
        if frequent_patterns:
            patterns, weights = zip(*frequent_patterns.items())
        else:
            patterns, weights = [], []
        self.pattern_set = PatternSet(patterns=patterns, weights=weights)

    @staticmethod
    def feature_value_generator(feature_values: dict[np.generic, np.ndarray]) -> dict[np.generic, Generator]:
        return {feature: (v for v in values) for feature, values in feature_values.items()}

    @staticmethod
    def centering(arr: np.ndarray) -> np.ndarray:
        unique_values = len(set(arr))
        if unique_values < len(arr):
            return rutils.cluster_centering(arr, max_clusters=unique_values)
        return rutils.bin_centering(arr)

    def group_continuous_thresholds(self):
        feature_values_leq = defaultdict(list)
        feature_values_gt = defaultdict(list)
        for path in self.paths:
            for node in path:
                # this is the prefix applied by our feature encoder
                if self.feature_names[node.feature].startswith("num__"):
                    if node.leq_threshold:
                        feature_values_leq[node.feature].append(node.threshold)
                    else:
                        feature_values_gt[node.feature].append(node.threshold)
    
        return feature_values_leq, feature_values_gt
    
    def discretize_continuous_thresholds(self):
        feature_values_leq, feature_values_gt = self.group_continuous_thresholds()
        return self.feature_value_generator({
            k: self.centering(np.array(v)) for k, v in feature_values_leq.items()
        }), self.feature_value_generator({
            k: self.centering(np.array(v)) for k, v in feature_values_gt.items()
        })
