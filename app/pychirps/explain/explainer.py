import numpy as np
import pandas as pd
from functools import cached_property
from app.pychirps.path_mining.forest_explorer import ForestExplorer
from app.pychirps.rule_mining.pattern_miner import RandomForestPatternMiner
from app.pychirps.rule_mining.rule_miner import RuleMiner, CounterfactualEvaluater
from app.pychirps.data_prep.pandas_encoder import PandasEncoder
from app.pychirps.model_prep.model_building import RandomForestClassifier


def predict(
    model: RandomForestClassifier,
    feature_frame: pd.DataFrame,
    encoder: PandasEncoder,
) -> np.ndarray:
    encoded_instance, _ = encoder.transform(features=feature_frame)
    return model.predict(encoded_instance)


class Explainer:
    """
    Wrapper class for mining rules from various tree ensemble methods.

    Attributes:
        best_pattern: The optimal pattern found during rule mining
        best_precision: Precision score of the best pattern
        best_coverage: Coverage score of the best pattern
        best_entropy: Entropy measure of the best pattern
        best_excl_cov: Exclusive coverage of the best pattern
        best_stability: Stability measure of the best pattern
    """

    def __init__(
        self,
        model: RandomForestClassifier,
        encoder: PandasEncoder,
        feature_frame: pd.DataFrame,
        prediction,
        min_support: float = 0.1,
        pruning_tolerance: float = 0.05,
    ):
        self.model = model
        self.prediction = prediction
        self.encoder = encoder

        self.forest_explorer = ForestExplorer(self.model, self.encoder)
        encoded_instance, _ = encoder.transform(features=feature_frame)
        encoded_instance = encoded_instance.astype(np.float32).reshape(1, -1)
        self.forest_path = self.forest_explorer.get_forest_path(encoded_instance)
        self.pattern_miner = RandomForestPatternMiner(
            forest_path=self.forest_path,
            feature_names=self.encoder.preprocessor.get_feature_names_out().tolist(),
            prediction=self.prediction,
            min_support=min_support,
        )

        transformed_features, transformed_targets = self.encoder.transform()
        preds = model.predict(transformed_features)
        classes = np.unique(transformed_targets)
        self.rule_miner = RuleMiner(
            pattern_miner=self.pattern_miner,
            y_pred=self.prediction,
            features=transformed_features,
            preds=preds,
            classes=classes,
            pruning_tolerance=pruning_tolerance,
        )

        self._counterfactual_evaluator = CounterfactualEvaluater(
            pattern=tuple(),
            y_pred=self.prediction,
            features=transformed_features,
            preds=preds,
            classes=classes,
        )

        # Initialize rule mining result attributes
        self.best_pattern = tuple()
        self.best_precision = 0
        self.best_coverage = None
        self.best_entropy = None
        self.best_excl_cov = None
        self.best_stability = None

    def _update_best_metrics_from_rule_miner(self):
        """Update best metrics from the rule miner's results."""
        self.best_pattern = self.rule_miner.best_pattern
        self.best_precision = self.rule_miner.best_precision
        self.best_coverage = self.rule_miner.best_coverage
        self.best_entropy = self.rule_miner.best_entropy
        self.best_excl_cov = self.rule_miner.best_excl_cov
        self.best_stability = self.rule_miner.best_stability

    def hill_climb(self):
        self.rule_miner.hill_climb()
        self.rule_miner.pruning_stage_two()
        self._update_best_metrics_from_rule_miner()

    @cached_property
    def counterfactual_evaluator(self):
        if self.best_pattern:
            self._counterfactual_evaluator.pattern = self.best_pattern
        return self._counterfactual_evaluator
