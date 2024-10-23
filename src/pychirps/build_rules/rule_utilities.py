import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class NodePattern:
    feature: int
    threshold: float
    leq_threshold: bool

    def __lt__(self, other: "NodePattern"):
        return self.feature < other.feature


def stability(
    y_pred: int, z_pred: np.ndarray, Z: np.ndarray, pattern: tuple[NodePattern], K: int
):
    # K should be the number of classes
    # a modified/Laplace corrected precision function to ensure rules do not overfit (i.e. they fit a single instance but none of its neighbours)
    rule_applies_indices = apply_rule(rule=pattern, Z=Z)
    # which instances have the same prediction as the instance we are trying to explain
    y_pred_indices = np.where(y_pred == z_pred)[0]
    # how many of the instances that satisfy the rule have the same prediction as the instance we are trying to explain
    correct_pred = len(set(rule_applies_indices).intersection(set(y_pred_indices)))
    # return the precision of the rule with a Laplace correction.
    # ASSUMPTION: x, the instance we are trying to explain is not in the set Z. If x is in Z, we would not + 1 the numerator
    return (correct_pred + 1) / (len(rule_applies_indices) + K)


def apply_rule(rule: tuple[NodePattern], Z: np.ndarray) -> np.ndarray:
    # Extract feature indices, thresholds, and leq_threshold flags
    features = np.array([node.feature for node in rule])
    thresholds = np.array([node.threshold for node in rule])
    leq_thresholds = np.array([node.leq_threshold for node in rule])

    # Get the feature values from Z
    feature_values = Z[:, features]

    # Apply conditions in a vectorized way
    rule_applies = np.ones(Z.shape[0], dtype=bool)
    # leq_thresholds is the less than or equal mask
    rule_applies &= np.all((feature_values <= thresholds)[:, leq_thresholds], axis=1)
    # ~leq_thresholds is the greater than mask
    rule_applies &= np.all((feature_values > thresholds)[:, ~leq_thresholds], axis=1)

    # Return the indices where the rule applies
    return np.where(rule_applies)[0]
