from src.pychirps.extract_paths.classification_trees import ForestPath
from pyfpgrowth import find_frequent_patterns
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class NodePattern:
    feature: int
    threshold: float
    leq_threshold: bool

    def __lt__(self, other: "NodePattern"):
        return self.feature < other.feature


class RuleMiner:
    def __init__(self, forest_path: ForestPath, prediction: Optional[int] = None):
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


#     def get_paths_by_prediction(self, prediction: int) -> list[list[NodePattern]]:

#         weighted_patterns = [
#             (
#                 node.path_weight,
#                 NodePattern(
#                     feature=node.feature,
#                     threshold=node.threshold,
#                     leq_threshold=node.leq_threshold,
#                 )
#             )
#             for gathered_path in forest_paths.gathered_paths
#             if gathered_path.prediction == prediction
#             for path in gathered_path.paths
#             for node in path
#         ]

#         # ensure support to an absolute number of instances rather than a fraction
#         if support <= 1:
#             support = round(support * len(self.paths))

#         # normalise the weights so min(weights) = 1.0
#         weighted_counts = np.round(self.paths_weights * 1/min(self.paths_weights)).astype('int')

#         # replicate the paths a number of times according to weighted counts
#         self.paths = list(chain.from_iterable(map(repeat, self.paths, weighted_counts)))

# def mine_rules(self, instance: np.ndarray, min_support: int) -> list:


# def mine_patterns(self, sample_instances=None, paths_lengths_threshold=2, support=0.1):

#     # repeat paths if max length > path length threshold
#     # e.g. for boosted models with stumps of depth 1 or 2, it doesn't make much sense
#     # for longer paths, the boosting weight is used to increase the support count
#     if len(max(self.paths, key=len)) >= paths_lengths_threshold:

#         # ensure support to an absolute number of instances rather than a fraction
#         if support <= 1:
#             support = round(support * len(self.paths))

#         # normalise the weights so min(weights) = 1.0
#         weighted_counts = np.round(self.paths_weights * 1/min(self.paths_weights)).astype('int')

#         # replicate the paths a number of times according to weighted counts
#         self.paths = list(chain.from_iterable(map(repeat, self.paths, weighted_counts)))

#         # FP mining
#         self.patterns = find_frequent_patterns(self.paths, support)
#         # normalise support score
#         self.patterns = {patt : self.patterns[patt]/len(self.paths) for patt in self.patterns}

#     # otherwise, convert paths to patterns giving weights as support
#     else:
#         # ensure support to a fraction
#         if support > 1:
#             support = support / len(self.paths)

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
