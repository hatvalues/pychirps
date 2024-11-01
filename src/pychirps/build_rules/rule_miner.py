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


#         entropy_weighted_patterns = defaultdict(np.float32)
#         instances, labels = self.init_instances(instances=sample_instances)
#         prior = p_count_corrected(labels, [i for i in range(len(self.class_names))])

#         # neutral estimator weights - SAMME.R
#         if np.all([i == 1.0 for i in self.paths_weights]):
#             # weight by how well it discriminates - how different from prior, based on kl-div
#             # assumiing we took majority or confidence weights, this is a test in the direction of the posterior
#             paths_weights = [contingency_test(ppp, prior['p_counts'], 'kldiv') for ppp in self.paths_pred_proba]
#         else:
#             # otherwise the weights from classic AdaBoost or SAMME, or the predicted value of the GBT model
#             paths_weights = self.paths_weights

#         for j, p in enumerate(self.paths):
#             items = []
#             kldivs = []
#             # collect
#             rule = [] # [item] otherwise when length is one it would iterate into a character list
#             current_kldiv = 0
#             for item in p:
#                 rule.append(item)
#                 idx = self.apply_rule(rule=rule, instances=instances, features=self.features_enc)
#                 p_counts = p_count_corrected(labels[idx], [i for i in range(len(self.class_names))])
#                 # collect the (conditional) information for each node in the tree/stump: how well does individual node discriminate? given node hierarchy
#                 kldiv = contingency_test(p_counts['counts'], prior['counts'], 'kldiv') - current_kldiv
#                 current_kldiv = kldiv
#                 kldivs.append(kldiv)
#             for e, item in zip(kldivs, p):
#                 # running sum of the normalised then tree-weighted entropy for any node found in the ensemble
#                 if sum(kldivs) * paths_weights[j] > 0: # avoid div by zero
#                     entropy_weighted_patterns[item] += e / sum(kldivs) * paths_weights[j]

#         # normalise the partial weighted entropy so it can be filtered by support (support takes on a slightly different meaning here)
#         if len(entropy_weighted_patterns) == 1: # freak case but can happen - and the MinMaxScaler will give 0 when fitted to a single value
#             entropy_weighted_patterns['dummy'] += 0.0
#         scaler = MinMaxScaler()
#         scaler.fit([[w] for w in dict(entropy_weighted_patterns).values()])
#         self.patterns = {((p), ) : scaler.transform([[w]])[0][0] for p, w in dict(entropy_weighted_patterns).items() \
#                                 if scaler.transform([[w]]) >= support }
