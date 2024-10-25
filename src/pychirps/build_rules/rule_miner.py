from pychirps.build_rules.pattern_miner import PatternMiner
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from functools import cached_property


class RuleMiner:
    def __init__(
        self, pattern_miner: PatternMiner, features: np.ndarray, preds: np.ndarray
    ):
        self._pattern_miner = pattern_miner
        self.features = features
        self.preds = preds

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

    def hill_climb(self):
        # hill climbing algorithm to find the best combination of patterns
        # start with the patterns sorted by their weights
        # for each pattern, add it to the rule set and evaluate the rule set
        # if the rule set is better than the previous rule set, keep the pattern
        # otherwise, remove the pattern from the rule set
        sorted_patterns = sorted(
            zip(self.patterns, self.weights), key=lambda x: x[1], reverse=True
        )
        pass

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
