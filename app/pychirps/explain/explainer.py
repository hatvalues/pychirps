import numpy as np
import pandas as pd
from functools import cached_property
from app.pychirps.path_mining.classification_trees import random_forest_paths_factory
from app.pychirps.path_mining.forest_explorer import ForestExplorer
from app.pychirps.rule_mining.pattern_miner import PatternMiner
from app.pychirps.rule_mining.rule_miner import RuleMiner, CounterfactualEvaluater
from app.pychirps.data_prep.pandas_encoder import PandasEncoder
from app.pychirps.model_prep.model_building import RandomForestClassifier


def predict(
    model: RandomForestClassifier,
    feature_frame: pd.DataFrame,
    dummy_target_class: str,
    encoder: PandasEncoder,
) -> np.ndarray:
    dummy_target = pd.Series(dummy_target_class)
    encoded_instance, _ = encoder.transform(feature_frame, dummy_target)
    return model.predict(encoded_instance)


class Explainer:
    def __init__(
        self,
        model: RandomForestClassifier,
        encoder: PandasEncoder,
        instance,
        prediction,
        min_support: float = 0.1,
        pruning_tolerance: float = 0.05,
    ):
        self.model = model
        self.prediction = prediction
        self.encoder = encoder

        self.forest_explorer = ForestExplorer(self.model, self.encoder)
        self.forest_path = random_forest_paths_factory(self.forest_explorer, instance)
        self.pattern_miner = PatternMiner(
            forest_path=self.forest_path,
            feature_names=self.encoder.preprocessor.get_feature_names_out().tolist(),
            prediction=self.prediction,
            min_support=min_support,
        )

        transformed_features, transformed_targets = self.encoder.transform()
        preds = model.predict(encoder.features)
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

        self.best_pattern = tuple()
        self.best_precision = 0

    def set_completion_properties(self):
        self.best_pattern = self.rule_miner.best_pattern
        self.best_precision = self.rule_miner.best_precision
        self.best_coverage = self.rule_miner.best_coverage
        self.best_entropy = self.rule_miner.best_entropy
        self.best_excl_cov = self.rule_miner.best_excl_cov
        self.best_stability = self.rule_miner.best_stability

    def hill_climb(self):
        self.rule_miner.hill_climb()
        self.rule_miner.pruning_stage_two()
        self.set_completion_properties()

    @cached_property
    def counterfactual_evaluator(self):
        if self.best_pattern:
            self._counterfactual_evaluator.pattern = self.best_pattern
        return self._counterfactual_evaluator
