from src.pychirps.extract_paths.classification_trees import ForestPath
from pyfpgrowth import find_frequent_patterns
from src.pychirps.build_rules.rule_utilities import NodePattern
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Callable
from src.pychirps.build_rules.rule_utilities import cluster_centering
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
        discretizing_function: Callable = cluster_centering,
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

        frequent_patterns = find_frequent_patterns(self.paths, self.support)
        if frequent_patterns:
            patterns, weights = zip(*frequent_patterns.items())
        else:
            patterns, weights = [], []
        self.pattern_set = PatternSet(patterns=patterns, weights=weights)

    def descretize_continuous_thresholds(self):
        feature_values_leq = defaultdict(list)
        feature_values_gt = defaultdict(list)
        for pattern in self.pattern_set.patterns:
            for node in pattern:
                # this is the prefix applied by our feature encoder
                if self.feature_names[node.feature].startswith("num__"):
                    if node.leq_threshold:
                        feature_values_leq[node.feature].append(node.threshold)
                    else:
                        feature_values_gt[node.feature].append(node.threshold)

        return {
            k: cluster_centering(np.array(v)) for k, v in feature_values_leq.items()
        }, {k: cluster_centering(np.array(v)) for k, v in feature_values_gt.items()}

    # def discretize_paths(self, bins=4, equal_counts=False, var_dict=None):
    #     # check if bins is not numeric or can't be cast, then force equal width (equal_counts = False)
    #     var_dict, _ = self.init_dicts(var_dict=var_dict)

    #     if equal_counts:
    #         def hist_func(x, bins, weights=None):
    #             npt = len(x)
    #             bns = np.quantile(x, [0.0, .25, .5, .75, 1.0])
    #             return(np.histogram(x, bns, weights=weights))
    #     else:
    #         def hist_func(x, bins, weights=None):
    #             return(np.histogram(x, bins, weights=weights))

    #         paths_discretized = []
    #         for nodes in self.paths:
    #             nodes_discretized = []
    #             for f, t, v in nodes:
    #                 if f == feature:
    #                     if t == False: # greater than, lower bound
    #                         v = lower_discretize(v)
    #                     else:
    #                         v = upper_discretize(v)
    #                 nodes_discretized.append((f, t, v))
    #             paths_discretized.append(nodes_discretized)
    #         # at the end of each loop, update the instance variable

    #         # descretised paths can result in duplicates items, which results in redundancy in the FP
    #         self.paths = [[]] * len(paths_discretized)
    #         for p, path in enumerate(paths_discretized):
    #             self.paths[p] = [i for i in set(path)]
