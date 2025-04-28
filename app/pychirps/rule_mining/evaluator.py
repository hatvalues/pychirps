from app.pychirps.rule_mining.rule_utilities import NodePattern
import app.pychirps.rule_mining.rule_utilities as rutils
from collections import Counter
import numpy as np


class Evaluator:
    def __init__(
        self,
        y_pred: np.uint8,
        features: np.ndarray,
        preds: np.ndarray,
        classes=np.array([0, 1], dtype=np.uint8),
    ):
        self.y_pred = y_pred
        self.features = features
        self.preds = preds
        self.classes = np.sort(classes)

    def entropy_score(self, pattern: tuple[NodePattern]) -> float:
        rule_applies_indices = rutils.apply_rule(pattern, self.features)
        rule_applies_preds = self.preds[rule_applies_indices]
        pred_count = Counter(rule_applies_preds)
        # ensuring same order each time
        pred_count.update({k: 0 for k in self.classes if k not in pred_count})
        return rutils.entropy(np.array([pred_count[cla] for cla in self.classes]))

    def stability_score(self, pattern: tuple[NodePattern]) -> float:
        return rutils.stability(
            y_pred=self.y_pred,
            z_pred=self.preds,
            Z=self.features,
            pattern=pattern,
            K=len(self.classes),
        )

    def exclusive_coverage_score(self, pattern: tuple[NodePattern]) -> float:
        return rutils.exclusive_coverage(
            y_pred=self.y_pred,
            z_pred=self.preds,
            Z=self.features,
            pattern=pattern,
            K=len(self.classes),
        )

    def precision_score(self, pattern: tuple[NodePattern]) -> float:
        return rutils.precision(
            y_pred=self.y_pred,
            z_pred=self.preds,
            Z=self.features,
            pattern=pattern,
        )

    def coverage_score(self, pattern: tuple[NodePattern]) -> float:
        return rutils.coverage(
            Z=self.features,
            pattern=pattern,
        )
