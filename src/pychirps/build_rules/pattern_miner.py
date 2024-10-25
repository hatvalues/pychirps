from src.pychirps.extract_paths.classification_trees import ForestPath
from pyfpgrowth import find_frequent_patterns
from src.pychirps.build_rules.rule_utilities import NodePattern
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PatternSet:
    patterns: list[tuple[NodePattern]]
    weights: np.ndarray


class PatternMiner:
    def __init__(
        self,
        forest_path: ForestPath,
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
