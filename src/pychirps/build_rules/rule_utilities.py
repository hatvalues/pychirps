import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class NodePattern:
    feature: int
    threshold: float
    leq_threshold: bool

    def __lt__(self, other: "NodePattern"):
        return self.feature < other.feature


def apply_rule(rule: tuple[NodePattern], Z: np.ndarray) -> np.ndarray:
    size = len(rule)
    features = np.empty(size, dtype=int)
    thresholds = np.empty(size, dtype=float)
    leq_thresholds = np.empty(size, dtype=bool)

    for i, node in enumerate(rule):
        features[i] = node.feature
        thresholds[i] = node.threshold
        leq_thresholds[i] = node.leq_threshold

    feature_values = Z[:, features]

    rule_applies = np.ones(Z.shape[0], dtype=bool)
    # leq_thresholds is the less than or equal mask
    rule_applies &= np.all((feature_values <= thresholds)[:, leq_thresholds], axis=1)
    # ~leq_thresholds is the greater than mask
    rule_applies &= np.all((feature_values > thresholds)[:, ~leq_thresholds], axis=1)

    # Return the indices where the rule applies
    return np.where(rule_applies)[0]


def stability(y_pred: int, z_pred: np.ndarray, Z: np.ndarray, pattern: tuple[NodePattern], K: int) -> float:
    # Stability is a modified/Laplace corrected precision function to ensure rules do not overfit (i.e. they fit a single instance but none of its neighbours)
    # K should be the number of classes. 1/K <= stability < 1, and we exclude the instance itself from the neighbours

    rule_applies_indices = apply_rule(rule=pattern, Z=Z)
    print(len(rule_applies_indices))

    y_pred_indices = np.where(y_pred == z_pred)[0]

    same_pred = len(set(rule_applies_indices).intersection(set(y_pred_indices)))
    print(same_pred)
    # ASSUMPTION: x, the instance we are trying to explain is not in the set Z. If x is in Z, we would not + 1 the numerator
    print(same_pred + 1)
    print(len(rule_applies_indices) + K)
    return (same_pred + 1) / (len(rule_applies_indices) + K)


def true_negative_rate(y_pred: int, z_pred: np.ndarray, Z: np.ndarray, pattern: tuple[NodePattern]) -> float:
    # True Negative Rate (TNR) is the proportion of instances in Z that are not covered by the rule and are predicted with another class, that is: 
    # numerator = not covered by rule and predicted with other class,
    # denominator = all instance (except instance itself) predicted with other class
    rule_applies_indices = apply_rule(rule=pattern, Z=Z)

    rule_not_apply_indices = set(np.indices(Z)) - set(rule_applies_indices)

    other_pred_indices = np.where(y_pred != z_pred)[0]

    return len(set(rule_not_apply_indices).intersection(set(other_pred_indices))) / len(other_pred_indices)


def exclusive_coverage(y_pred: int, z_pred: np.ndarray, Z: np.ndarray, pattern: tuple[NodePattern], K: int) -> float:
    # Exclusive coverage is the Lplace corrected proportion of instances in Z that are covered by the rule, excluding the instance itself from its neighbours...
    # ... multiplied by the True Negative Rate (TNR) of the rule
    rule_applies_indices = apply_rule(rule=pattern, Z=Z)
    # ASSUMPTION: x, the instance we are trying to explain is not in the set Z. If x is in Z, we would not + 1 the numerator
    return len(rule_applies_indices + 1) / (len(Z) + K) * true_negative_rate(y_pred=y_pred, z_pred=z_pred, Z=Z, pattern=pattern)
