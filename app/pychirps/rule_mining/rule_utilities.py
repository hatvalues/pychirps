import warnings
import numpy as np
from dataclasses import dataclass
from sklearn.cluster import KMeans
from scipy.stats import entropy as scipy_entropy
from scipy.stats import wasserstein_distance
from sklearn.metrics import silhouette_score
from app.config import DEFAULT_RANDOM_SEED
from ordered_set import OrderedSet


@dataclass(frozen=True)
class NodePattern:
    feature: np.uint8
    threshold: float
    leq_threshold: bool

    def __lt__(self, other: "NodePattern"):
        return self.feature < other.feature


def apply_rule(rule: tuple[NodePattern], Z: np.ndarray) -> np.ndarray:
    # Apply a rule to a dataset Z and return the indices where the rule applies
    # The rule is formed by a tuple of NodePattern objects
    # For each NodePattern, the rule applies if the feature value is less than or equal to the threshold if leq_threshold is True,
    # or greater than the threshold if leq_threshold is False
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


def precision(
    y_pred: np.uint8,
    z_pred: np.ndarray,
    Z: np.ndarray,
    pattern: tuple[NodePattern],
) -> float:
    # Precision is the proportion of instances in Z that are covered by the rule and are predicted with the same class, that is:
    # numerator = covered by rule and predicted with same class,
    # denominator = all instance (except instance itself) predicted with same class
    rule_applies_indices = apply_rule(rule=pattern, Z=Z)
    rule_applies_count = len(rule_applies_indices)

    y_pred_indices = np.where(y_pred == z_pred)[0]

    same_pred = len(set(rule_applies_indices).intersection(set(y_pred_indices)))
    return same_pred / rule_applies_count if rule_applies_count > 0.0 else 0.0


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
    # ASSUMPTION: x, the instance we are trying to explain is not in the set Z. If x is in Z, we would not + 1 the numerator
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


def coverage(Z: np.ndarray, pattern: tuple[NodePattern]) -> float:
    # Coverage is the proportion of instances in Z that are covered by the rule, that is:
    # numerator = covered by rule,
    # denominator = all instance (except instance itself)
    rule_applies_indices = apply_rule(rule=pattern, Z=Z)
    return len(rule_applies_indices) / len(Z)


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
        # THESIS CHAPTER 6: Equation 6.1 right-hand term
    return (cardinality - cardinality_regularizing_weight) / cardinality


def objective_function(
    stability_score: float,
    excl_cov_score: float,
    cardinality: int,
    blending_weight: float,
    cardinality_regularizing_weight,
) -> float:
    # objective function to evaluate the quality of the rules
    return (
        (blending_weight * stability_score)
        + ((1 - blending_weight) * excl_cov_score)
        - adjusted_cardinality_weight(
            cardinality=cardinality,
            cardinality_regularizing_weight=cardinality_regularizing_weight,
        )
    )


# For the avoidance of confusion elsewhere in code, I'm separating the overloading of the scipy.entropy function
def entropy(p: np.ndarray) -> np.float64:
    return scipy_entropy(p)


def count_data_check(arr: np.ndarray) -> None:
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


def bin_centering(arr: np.ndarray) -> np.ndarray:
    bin_counts, auto_edges = np.histogram(arr, bins="auto")
    bin_sum_totals, bin_edges = np.histogram(
        arr, bins=len(auto_edges) - 1, weights=arr
    )  # weight each point by it's own value
    assert np.array_equal(auto_edges, bin_edges), "Drived Bins not equal to Auto Bins"
    bin_means = bin_sum_totals / bin_counts
    bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
    assert len(bin_means) == len(
        bin_midpoints
    ), "Different number of bin means and bin medians"
    bin_centres = np.where(np.isnan(bin_means), bin_midpoints, bin_means)
    distances = np.abs(arr[:, None] - bin_centres)
    nearest_midpoint_indices = np.argmin(distances, axis=1)
    centred = bin_midpoints[nearest_midpoint_indices]
    assert len(centred) == len(arr), "Output array length not equal input array length"
    return centred


def cluster_centering(
    arr: np.ndarray, min_clusters: np.uint8 = 2, max_clusters: np.uint8 = 8
) -> np.ndarray:
    if arr.size == 0:
        return np.array([])
    if np.allclose(arr, arr[0]):
        return arr
    best_score = -1
    collected_k_means = {}
    best_k = min_clusters
    for k in range(min_clusters, max_clusters + 1):
        collected_k_means[k] = KMeans(
            n_clusters=k, random_state=DEFAULT_RANDOM_SEED
        ).fit(arr.reshape(-1, 1))
        score = silhouette_score(arr.reshape(-1, 1), collected_k_means[k].labels_)
        if score > best_score:
            best_score = score
            best_k = k
    clusters = collected_k_means[best_k].labels_
    cluster_centers = collected_k_means[best_k].cluster_centers_
    centred = np.array([cluster_centers[cluster][0] for cluster in clusters])

    return centred


def merge_patterns(
    base_pattern: tuple[NodePattern], add_pattern: tuple[NodePattern]
) -> tuple[NodePattern]:
    return tuple(node for node in OrderedSet(base_pattern) | OrderedSet(add_pattern))


def pattern_covers_pattern(
    covering_pattern: tuple[NodePattern], test_pattern: tuple[NodePattern]
) -> bool:
    return set(covering_pattern).issuperset(set(test_pattern))
