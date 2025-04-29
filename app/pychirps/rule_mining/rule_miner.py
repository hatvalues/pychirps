from app.pychirps.rule_mining.evaluator import Evaluator
from app.pychirps.rule_mining.rule_utilities import NodePattern
import app.pychirps.rule_mining.rule_utilities as rutils
import numpy as np


class RuleMiner(Evaluator):
    def __init__(
        self,
        y_pred: np.uint8,
        features: np.ndarray,
        preds: np.ndarray,
        classes: np.ndarray,
        patterns: tuple[tuple[NodePattern]],
        cardinality_regularizing_weight: float = 0.5,
        pruning_tolerance: float = 0.05,
    ):
        super().__init__(y_pred, features, preds, classes)
        self.patterns = patterns
        self.cardinality_regularizing_weight = cardinality_regularizing_weight
        self.best_pattern = tuple()

    @staticmethod
    def prune_covered_patterns(
        covering_pattern: tuple[NodePattern], patterns: tuple[tuple[NodePattern]]
    ) -> tuple[tuple[NodePattern]]:
        return tuple(
            pattern
            for pattern in patterns
            if not rutils.pattern_covers_pattern(covering_pattern, pattern)
        )

    def hill_climb(
        self, blending_weight: float = 1.0, cardinality_regularizing_weight: float = 0.0
    ):
        # default input use naive hill climb with the objective function
        # hill climbing algorithm to find the best combination of patterns
        # start with the patterns sorted by their weights
        # pick the rule from the top and add it to the rule
        # if the rule set is better than the previous rule set, keep the pattern
        # loop until no more increase in objective function, no more patterns
        patterns = list(self.patterns)
        best_pattern = tuple()
        best_score = -np.inf
        while patterns:
            add_pattern = patterns.pop(0)
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

            # prune the input list of patterns to shorten the search list
            # output of pruning function is a tuple because we need hashable objects under the hood
            patterns = list(
                self.prune_covered_patterns(
                    covering_pattern=add_pattern, patterns=patterns
                )
            )

        self.best_pattern = best_pattern
        self.best_entropy = self.entropy_score(best_pattern)
        self.best_stability = self.stability_score(best_pattern)
        self.best_excl_cov = self.exclusive_coverage_score(best_pattern)
        self.best_precision = self.precision_score(best_pattern)
        self.best_coverage = self.coverage_score(best_pattern)

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

    def pruning_stage_two(self):
        # stage two of the pruning algorithm
        pass


class CounterfactualEvaluater(Evaluator):
    def __init__(
        self,
        pattern: tuple[NodePattern],
        y_pred: np.uint8,
        features: np.ndarray,
        preds: np.ndarray,
        classes=np.array([0, 1], dtype=np.uint8),
    ):
        super().__init__(y_pred, features, preds, classes)
        self.pattern = pattern

    @staticmethod
    def flip_node_pattern(node_pattern: NodePattern) -> NodePattern:
        return NodePattern(
            feature=node_pattern.feature,
            threshold=node_pattern.threshold,
            leq_threshold=not node_pattern.leq_threshold,
        )

    def get_counterfactuals(self) -> tuple[tuple[NodePattern]]:
        return tuple(
            tuple(
                self.pattern[:n]
                + (CounterfactualEvaluater.flip_node_pattern(node),)
                + self.pattern[n + 1 :]
            )
            for n, node in enumerate(self.pattern)
        )

    def evaluate_counterfactuals(self) -> tuple[float]:
        return tuple(
            (
                self.coverage_score(pattern=counterfactual),
                self.precision_score(pattern=counterfactual),
            )
            for counterfactual in self.get_counterfactuals()
        )
