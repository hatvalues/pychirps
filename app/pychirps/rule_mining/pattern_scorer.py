from app.pychirps.rule_mining.evaluator import Evaluator
from app.pychirps.rule_mining.rule_utilities import NodePattern
import app.pychirps.rule_mining.rule_utilities as rutils
from collections import defaultdict
from abc import ABC, abstractmethod
from functools import cached_property
import numpy as np


class PatternScorer(Evaluator, ABC):
    def __init__(
        self,
        patterns: tuple[tuple[NodePattern]],
        weights: np.ndarray,
        y_pred: np.uint8,
        features: np.ndarray,
        preds: np.ndarray,
        classes: np.ndarray,
        cardinality_regularizing_weight: float = 0.5,
    ):
        super().__init__(y_pred, features, preds, classes)
        self.patterns = patterns
        self._weights = weights
        self.cardinality_regularizing_weight = cardinality_regularizing_weight

    @cached_property
    def weights(self):
        min_weight = min(self._weights)
        max_weight = max(self._weights)
        if min_weight == max_weight:
            return np.ones(len(self._weights))
        else:
            return np.array([w / max_weight for w in self._weights])

    @cached_property
    def entropy_regularizing_weights(self):
        entropy_regularizing_weights = np.zeros(len(self.weights))
        for p, pattern in enumerate(self.patterns):
            entropy_regularizing_weights[p] = 1 - self.entropy_score(pattern)
        return entropy_regularizing_weights

    @abstractmethod
    @cached_property
    def custom_sorted_patterns(self):
        """
        Sorts the patterns based on a custom scoring function.
        The sorting is done in descending order, so the most important patterns come first.
        """
        pass


class RandomForestPatternScorer(PatternScorer):
    def __init__(
        self,
        patterns: tuple[tuple[NodePattern]],
        weights: np.ndarray,
        y_pred: np.uint8,
        features: np.ndarray,
        preds: np.ndarray,
        classes=np.array([0, 1], dtype=np.uint8),
        cardinality_regularizing_weight: float = 0.5,
    ):
        super().__init__(
            patterns,
            weights,
            y_pred,
            features,
            preds,
            classes,
            cardinality_regularizing_weight,
        )

    def pattern_importance_score(
        self,
        cardinality: np.uint8,
        support_regularizing_weight: float = 1.0,
        entropy_regularizing_weight: float = 1.0,
    ):
        # THESIS CHAPTER 6: Equation 6.1 full equation for scoring path segments extracted by frequent pattern mining
        # cardinality_regularizing_weight is the alpha parameter in the thesis
        # support_regularizing_weight is the sup parameter in the thesis
        # entropy_regularizing_weight is the w parameter in the thesis
        return (
            support_regularizing_weight
            * entropy_regularizing_weight
            * rutils.adjusted_cardinality_weight(
                cardinality=cardinality,
                cardinality_regularizing_weight=self.cardinality_regularizing_weight,
            )
        )

    @cached_property
    def custom_sorted_patterns(self):
        sorted_pattern_weights = sorted(
            zip(self.patterns, self.weights, self.entropy_regularizing_weights),
            key=lambda x: self.pattern_importance_score(
                cardinality=len(x[0]),
                support_regularizing_weight=x[1],
                entropy_regularizing_weight=x[2],
            ),
            reverse=True,
        )
        # these are now sorted by how they perform:
        # A. on the data set individually increasing entropy (higher is better)
        # B. how much support they receive (more frequent is better)
        # C. the cardinality of the pattern, (longer is better, more interaction terms)
        return tuple(pattern for pattern, _, _ in sorted_pattern_weights)


class AdaboostPatternScorer(PatternScorer):
    def __init__(
        self,
        patterns: tuple[tuple[NodePattern]],
        weights: np.ndarray,
        y_pred: np.uint8,
        features: np.ndarray,
        preds: np.ndarray,
        classes=np.array([0, 1], dtype=np.uint8),
        cardinality_regularizing_weight: float = 0.5,
    ):
        super().__init__(
            patterns,
            weights,
            y_pred,
            features,
            preds,
            classes,
            cardinality_regularizing_weight,
        )

    @cached_property
    def disagregated_patterns(self):
        # THESIS CHAPTER 7: for level of node in the pattern, get the entropy score
        # such that the first node's entropy score is just for that node
        # the second node's entropy score is for that node and the first node, and so on
        # normalize these weights so that the sum to the original weight
        # put each individual node into the map, accumulating the weights found.
        node_weight_map = defaultdict(float)
        for pattern, weight, e_weight in zip(
            self.patterns, self.weights, self.entropy_regularizing_weights
        ):
            cardinality = len(pattern)
            rule = tuple()
            entropy_scores = np.zeros(len(pattern))
            for n, node in enumerate(pattern):
                if n + 1 == cardinality:
                    entropy_scores[n] = (
                        e_weight  # last node, we have them precalculated so it's a short cut
                    )
                else:
                    rule = rule + (node,)
                    entropy_scores[n] = self.entropy_score(rule)
            # normalize the entropy scores
            entropy_scores = entropy_scores / np.sum(entropy_scores) * weight
            # accumulate the weights
            for n, node in enumerate(pattern):
                node_weight_map[node] += entropy_scores[n]

        return node_weight_map

    @cached_property
    def custom_sorted_patterns(self):
        # return individual nodes (keys) sorted by their score (values)
        return sorted(
            self.disagregated_patterns,
            key=self.disagregated_patterns.get,
            reverse=True,
        )
