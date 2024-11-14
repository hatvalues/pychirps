from app.pychirps.rule_mining.pattern_miner import PatternMiner
from app.pychirps.rule_mining.rule_utilities import NodePattern
import app.pychirps.rule_mining.rule_utilities as rutils
from sklearn.preprocessing import MinMaxScaler
from functools import cached_property
from collections import Counter
from typing import Callable
import numpy as np


class RuleMiner:
    def __init__(
        self,
        pattern_miner: PatternMiner,
        y_pred: np.uint8,
        features: np.ndarray,
        preds: np.ndarray,
        classes=np.array([0, 1], dtype=np.uint8),
        cardinality_regularizing_weight: float = 0.5,
    ):
        self._pattern_miner = pattern_miner
        self.y_pred = y_pred
        self.features = features
        self.preds = preds
        self.classes = np.sort(classes)
        self.cardinality_regularizing_weight = cardinality_regularizing_weight

    @property
    def patterns(self):
        return self._pattern_miner.pattern_set.patterns

    @cached_property
    def weights(self):
        min_weight = min(self._pattern_miner.pattern_set.weights)
        max_weight = max(self._pattern_miner.pattern_set.weights)
        if min_weight == max_weight:
            return np.ones(len(self._pattern_miner.pattern_set.weights))
        elif min_weight >= 1.0:
            feature_limit = min_weight / max_weight
            scaler = MinMaxScaler(feature_range=(feature_limit, 1 - feature_limit))
            return scaler.fit_transform(
                np.array(self._pattern_miner.pattern_set.weights).reshape(-1, 1)
            ).flatten()
        else:
            return np.array(self._pattern_miner.pattern_set.weights)

    def entropy_score(self, pattern: tuple[NodePattern]) -> float:
        rule_applies_indices = rutils.apply_rule(pattern, self.features)
        rule_applies_preds = self.preds[rule_applies_indices]
        pred_count = Counter(rule_applies_preds)
        # ensuring same order each time
        pred_count.update({k: 0 for k in self.classes if k not in pred_count})
        return rutils.entropy(np.array([pred_count[cla] for cla in self.classes]))

    @cached_property
    def entropy_regularizing_weights(self):
        entropy_regularizing_weights = np.zeros(len(self.weights))
        for p, pattern in enumerate(self.patterns):
            entropy_regularizing_weights[p] = 1 - self.entropy_score(pattern)
        return entropy_regularizing_weights

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

    def stability_score(self, pattern: tuple[NodePattern]) -> float:
        return rutils.stability(
            y_pred=self.y_pred,
            z_pred=self.preds,
            Z=self.features,
            pattern=pattern,
            K=len(self.classes),
        )

    def exclusive_coverage_score(self, pattern: tuple[NodePattern]) -> float:
        return rutils.exclusive_coverage(
            y_pred=self.y_pred,
            z_pred=self.preds,
            Z=self.features,
            pattern=pattern,
            K=len(self.classes),
        )

    def hill_climb(
        self, blending_weight: float = 1.0, cardinality_regularizing_weight: float = 0.0
    ):
        # default input use naive hill climb with the objective function
        # hill climbing algorithm to find the best combination of patterns
        # start with the patterns sorted by their weights
        # pick the rule from the top and add it to the rule
        # if the rule set is better than the previous rule set, keep the pattern
        # loop until no more increase in objective functtion, no more patterns
        sorted_patterns = [pattern for pattern in self.custom_sorted_patterns]
        best_pattern = tuple()
        best_score = -np.inf
        while sorted_patterns:
            add_pattern = sorted_patterns.pop(0)
            try_pattern = rutils.merge_patterns(best_pattern, add_pattern)
            try_stability = self.stability_score(try_pattern)
            try_excl_cov = self.exclusive_coverage_score(try_pattern)
            try_score = rutils.objective_function(
                stability_score=try_stability,
                excl_cov_score=try_excl_cov,
                cardinality=len(try_pattern),
                blending_weight=blending_weight,
                cardinality_regularizing_weight=cardinality_regularizing_weight,
            )
            if try_score > best_score:
                best_pattern = try_pattern
                best_score = try_score

        self.best_pattern = best_pattern
        self.best_stability = self.stability_score(best_pattern)
        self.best_excl_cov = self.exclusive_coverage_score(best_pattern)

    def dynamic_programming(self):
        # dynamic programming algorithm to find the best combination of patterns
        pass

    def genetic_algorithm(self):
        # genetic algorithm to find the best combination of patterns
        pass

    def beam_search(self):
        # beam search algorithm to find the best combination of patterns
        pass

    def tabu_search(self):
        # tabu search algorithm to find the best combination of patterns
        pass

    def simulated_annealing(self):
        # simulated annealing algorithm to find the best combination of patterns
        pass

        # will need to normalise the weights so min(weights) = 1.0
        # weighted_counts = np.round(self.paths_weights * 1/min(self.paths_weights)).astype('int')
