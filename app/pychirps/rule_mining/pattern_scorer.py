from app.pychirps.rule_mining.evaluator import Evaluator
from app.pychirps.rule_mining.rule_utilities import NodePattern
import app.pychirps.rule_mining.rule_utilities as rutils
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

class AdaboostPatternScorer(Evaluator):

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
    def custom_sorted_patterns(self):
        sorted_pattern_weights = sorted(
            zip(self.patterns, self.weights, self.entropy_regularizing_weights),
            key=lambda x: rutils.pattern_importance_score(
                cardinality=len(x[0]),
                support_regularizing_weight=x[1],
                entropy_regularizing_weight=x[2],
                cardinality_regularizing_weight=self.cardinality_regularizing_weight,
            ),
            reverse=True,
        )
        # these are now sorted by how they perform:
        # A. on the data set individually increasing entropy (higher is better)
        # B. how much support they receive (more frequent is better)
        # C. the cardinality of the pattern, (longer is better, more interaction terms)
        return tuple(pattern for pattern, _, _ in sorted_pattern_weights)