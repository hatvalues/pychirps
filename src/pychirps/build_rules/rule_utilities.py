import numpy as np
from scipy.stats import entropy as scipy_entropy
from scipy.stats import wasserstein_distance
from dataclasses import dataclass
import warnings


@dataclass(frozen=True)
class NodePattern:
    feature: np.uint8
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


def stability(
    y_pred: np.uint8,
    z_pred: np.ndarray,
    Z: np.ndarray,
    pattern: tuple[NodePattern],
    K: np.uint8,
) -> float:
    # Stability is a modified/Laplace corrected precision function to ensure rules do not overfit (i.e. they fit a single instance but none of its neighbours)
    # K should be the number of classes. 1/K <= stability < 1, and we exclude the instance itself from the neighbours

    rule_applies_indices = apply_rule(rule=pattern, Z=Z)

    y_pred_indices = np.where(y_pred == z_pred)[0]

    same_pred = len(set(rule_applies_indices).intersection(set(y_pred_indices)))
    print(same_pred)
    # ASSUMPTION: x, the instance we are trying to explain is not in the set Z. If x is in Z, we would not + 1 the numerator
    print(same_pred + 1)
    print(len(rule_applies_indices) + K)
    return (same_pred + 1) / (len(rule_applies_indices) + K)


def true_negative_rate(
    y_pred: np.uint8, z_pred: np.ndarray, Z: np.ndarray, pattern: tuple[NodePattern]
) -> float:
    # True Negative Rate (TNR) is the proportion of instances in Z that are not covered by the rule and are predicted with another class, that is:
    # numerator = not covered by rule and predicted with other class,
    # denominator = all instance (except instance itself) predicted with other class
    rule_applies_indices = apply_rule(rule=pattern, Z=Z)
    rule_does_not_apply_indices = set(range(len(Z))) - set(rule_applies_indices)
    other_pred_indices = np.where(y_pred != z_pred)[0]
    return len(
        set(rule_does_not_apply_indices).intersection(set(other_pred_indices))
    ) / len(other_pred_indices)


def exclusive_coverage(
    y_pred: np.uint8,
    z_pred: np.ndarray,
    Z: np.ndarray,
    pattern: tuple[NodePattern],
    K: np.uint8,
) -> float:
    # Exclusive coverage is the Lplace corrected proportion of instances in Z that are covered by the rule, excluding the instance itself from its neighbours...
    # ... multiplied by the True Negative Rate (TNR) of the rule
    rule_applies_indices = apply_rule(rule=pattern, Z=Z)
    print(len(rule_applies_indices) + 1)
    print(len(Z) + K)
    print(true_negative_rate(y_pred=y_pred, z_pred=z_pred, Z=Z, pattern=pattern))
    # ASSUMPTION: x, the instance we are trying to explain is not in the set Z. If x is in Z, we would not + 1 the numerator
    return (
        len(rule_applies_indices + 1)
        / (len(Z) + K)
        * true_negative_rate(y_pred=y_pred, z_pred=z_pred, Z=Z, pattern=pattern)
    )


def adjusted_cardinality_weight(
    cardinality: np.uint8, cardinality_regularizing_weight: float
):
    if cardinality_regularizing_weight < 0:
        warnings.warn(
            "Negative values of alpha are allowed but favour shorter path segments, which reduces the importance of interation terms."
        )
    return (cardinality - cardinality_regularizing_weight) / cardinality


def pattern_importance_score(
    cardinality: np.uint8,
    support_regularizing_weight: float = 1.0,
    entropy_regularizing_weight: float = 1.0,
    cardinality_regularizing_weight: float = 1.0,
):
    return (
        support_regularizing_weight
        * entropy_regularizing_weight
        * adjusted_cardinality_weight(
            cardinality=cardinality,
            cardinality_regularizing_weight=cardinality_regularizing_weight,
        )
    )


# For the avoidance of confusion elsewhere in code, I'm separating the overloading of the scipy.entropy function
def entropy(p: np.ndarray):
    return scipy_entropy(p)


def count_data_check(arr):
    if np.issubdtype(arr.dtype, np.floating):
        warnings.warn(
            "kl_div will work with probability distributions but if you're seeing this warning then you haven't implemented something properly for generalizing to other distance functions."
        )


# TODO: in most use cases in this package, we should be testing KL-div only on distributions where the pred class has a higher probability in Q than P. May need some guard rails for this.
def kl_div(p: np.ndarray, q: np.ndarray) -> np.float64:
    """Convenience wrapper for kl-div calculation, handling zero and smoothing so we never always get a valid value, never np.inf"""
    count_data_check(p)
    alpha = np.finfo(dtype="float16").eps
    p_smooth = np.random.uniform(size=len(p))
    q_smooth = np.random.uniform(
        size=len(p)
    )  # convex smooth idea https://mathoverflow.net/questions/72668/how-to-compute-kl-divergence-when-pmf-contains-0s
    p_smoothed = p_smooth * alpha + np.array(p) * (1 - alpha)
    q_smoothed = q_smooth * alpha + np.array(q) * (1 - alpha)
    # NOTE: scipy_entropy normalizes inputs by default
    return scipy_entropy(p_smoothed, q_smoothed)


def ws_dis(p: np.ndarray, q: np.ndarray) -> np.float64:
    """Adjusted Wasserstein Distance, normalized by total count."""
    count_data_check(p)
    return wasserstein_distance(p, q) / p.sum()
