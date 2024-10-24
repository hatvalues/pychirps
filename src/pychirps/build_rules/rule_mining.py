from src.pychirps.extract_paths.classification_trees import ForestPath
from pyfpgrowth import find_frequent_patterns
from src.pychirps.build_rules.rule_utilities import NodePattern
import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class PatternSet:
    patterns: list[tuple[NodePattern]]
    weights: list[float]

    def __post_init__(self):
        if len(self.patterns) != len(self.weights):
            raise ValueError("Patterns and weights must be the same length")
        if len(self.weights):
            min_weight = np.min(self.weights)
            if min_weight == 1:
                max_weight = np.max(self.weights)
                self.weights = [weight / max_weight for weight in self.weights]


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


class RuleMiner:
    def __init__(self, patterns: dict[tuple[NodePattern], float]):
        self.patterns = patterns,
    

    def hill_climb(self):
        # hill climbing algorithm to find the best combination of patterns
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
