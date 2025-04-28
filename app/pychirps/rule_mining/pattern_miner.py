from app.pychirps.path_mining.forest_explorer import ForestPath
from pyfpgrowth import find_frequent_patterns
from app.pychirps.rule_mining.rule_utilities import NodePattern
from collections import defaultdict
from dataclasses import dataclass
from typing import Generator
import app.pychirps.rule_mining.rule_utilities as rutils
import numpy as np


@dataclass(frozen=True)
class PatternSet:
    patterns: list[tuple[NodePattern]]
    weights: np.ndarray


class PatternMiner:
    def __init__(
        self,
        forest_path: ForestPath,
        feature_names: list[str],
        prediction: np.uint8,
        min_support: np.float16
    ):
        if min_support > 1.0:
            raise ValueError("Set min_support using a fraction")
        self.support = round(min_support * len(forest_path.paths))
        self.prediction = prediction
        self.feature_names = feature_names
        prediction_paths = forest_path.get_paths_for_prediction(prediction)
        if not prediction_paths:
            raise ValueError(f"No paths found for prediction {prediction}")
        # weights are always 1.0 for RF but not for Adaboost
        tree_node_paths, self.weights = zip(*prediction_paths) # transposes, unpacking
        # tree_nodepaths simplify to node patterns
        self.paths = tuple(
            tuple(
                NodePattern(
                    feature=node.feature,
                    threshold=node.threshold,
                    leq_threshold=node.leq_threshold,
                )
                for node in tree_node_path
            )
            for tree_node_path in tree_node_paths
        )

        feature_values_leq, feature_values_gt = self.discretize_continuous_thresholds()
        discretized_paths = []
        for path in self.paths:
            nodes = []
            for node in path:
                if (
                    self.feature_names[node.feature].startswith("num__")
                    and node.leq_threshold
                ):
                    nodes.append(
                        NodePattern(
                            feature=node.feature,
                            threshold=next(feature_values_leq[node.feature]),
                            leq_threshold=True,
                        )
                    )
                elif (
                    self.feature_names[node.feature].startswith("num__")
                    and not node.leq_threshold
                ):
                    nodes.append(
                        NodePattern(
                            feature=node.feature,
                            threshold=next(feature_values_gt[node.feature]),
                            leq_threshold=False,
                        )
                    )
                else:
                    nodes.append(node)
            discretized_paths.append(nodes)

        self.discretized_paths = tuple(
            tuple(node for node in nodes) for nodes in discretized_paths
        )

    @staticmethod
    def feature_value_generator(
        feature_values: dict[np.generic, np.ndarray],
    ) -> dict[np.generic, Generator]:
        return {
            feature: (v for v in values) for feature, values in feature_values.items()
        }

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
        return self.feature_value_generator(
            {k: self.centering(np.array(v)) for k, v in feature_values_leq.items()}
        ), self.feature_value_generator(
            {k: self.centering(np.array(v)) for k, v in feature_values_gt.items()}
        )


class RandomForestPatternMiner(PatternMiner):
    def __init__(
        self,
        forest_path: ForestPath,
        feature_names: list[str],
        prediction: np.uint8,
        min_support: np.float16 = 0.1,
    ):
        super().__init__(forest_path, feature_names, prediction, min_support)

        frequent_patterns = find_frequent_patterns(self.discretized_paths, self.support)
        if frequent_patterns:
            self.patterns, self.weights = zip(*frequent_patterns.items())
        else:
            self.patterns, self.weights = tuple(), tuple()


class AdaboostPatternMiner(PatternMiner):
    def __init__(
        self,
        forest_path: ForestPath,
        feature_names: list[str],
        prediction: np.uint8,
        min_support: np.float16 = 0.1,
    ):
        super().__init__(forest_path, feature_names, prediction, min_support)
        self.support = round(min_support * len(forest_path.paths))
        if self.discretized_paths:
            self.patterns, self.weights = self.discretized_paths, self.weights
        else:
            self.patterns, self.weights = tuple(), tuple()