from src.pychirps.build_rules.pattern_miner import PatternMiner
import src.pychirps.build_rules.rule_utilities as rutils
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from functools import cached_property
from collections import Counter
from typing import Callable


class RuleMiner:
    def __init__(
        self,
        pattern_miner: PatternMiner,
        y_pred: np.uint8,
        features: np.ndarray,
        preds: np.ndarray,
        classes=np.array([0, 1], dtype=np.uint8),
        cardinality_regularizing_weight: float = 0.5,
        entropy_function: Callable = rutils.entropy,
    ):
        self._pattern_miner = pattern_miner
        self.y_pred = y_pred
        self.features = features
        self.preds = preds
        self.classes = np.sort(classes)
        self.cardinality_regularizing_weight = cardinality_regularizing_weight
        self.entropy_function = entropy_function

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

    @cached_property
    def entropy_regularizing_weights(self):
        entropy_regularizing_weights = np.zeros(len(self.weights))
        for p, pattern in enumerate(self.patterns):
            rule_applies_indices = rutils.apply_rule(pattern, self.features)
            rule_applies_preds = self.preds[rule_applies_indices]
            pred_count = Counter(rule_applies_preds)
            pred_count.update({k: 0 for k in self.classes if k not in pred_count})
            entropy_regularizing_weights[p] = (
                # ensuring same order each time
                self.entropy_function(
                    np.array([pred_count[cla] for cla in self.classes])
                )
            )
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

    def hill_climb(self):
        # hill climbing algorithm to find the best combination of patterns
        # start with the patterns sorted by their weights
        # pick the rule from the top and add it to the rule
        # in the case of tied weights, pick the one with highest stability
        # in the case of tied stability, pick the one with highest exclusive coverage
        # compare current stability with previous stability
        # if the rule set is better than the previous rule set, keep the pattern
        # prune duplicate nodes
        # otherwise, remove the pattern from the rule set
        # loop until no more stability increase, no more patterns, or rule reaches max length
        sorted_patterns = self.custom_sorted_patterns
        print([len(p) for p in sorted_patterns])

    def dynamic_programming(self):
        # dynamic programming algorithm to find the best combination of patterns
        pass

    def genetic_algorithm(self):
        # genetic algorithm to find the best combination of patterns
        pass

    def beam_search(self):
        # beam search algorithm to find the best combination of patterns
        pass

    def objective_function(self):
        # objective function to evaluate the quality of the rules
        pass

    def tabu_search(self):
        # tabu search algorithm to find the best combination of patterns
        pass

    def simulated_annealing(self):
        # simulated annealing algorithm to find the best combination of patterns
        pass

        # will need to normalise the weights so min(weights) = 1.0
        # weighted_counts = np.round(self.paths_weights * 1/min(self.paths_weights)).astype('int')
