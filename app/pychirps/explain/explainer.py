import numpy as np
from functools import cached_property
from app.pychirps.path_mining.classification_trees import random_forest_paths_factory
from app.pychirps.path_mining.forest_explorer import ForestExplorer
from app.pychirps.rule_mining.pattern_miner import PatternMiner
from app.pychirps.rule_mining.rule_miner import RuleMiner, CounterfactualEvaluater


class Explainer:
    def __init__(self, model, encoder, instance, prediction, min_support: float = 0.1):
        self.model = model
        self.prediction = prediction
        self.encoder = encoder

        self.forest_explorer = ForestExplorer(self.model, self.encoder)
        instance32 = instance.to_numpy().astype(np.float32).reshape(1, -1)
        self.forest_path = random_forest_paths_factory(self.forest_explorer, instance32)
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
        )

        self._counterfactual_evaluator = CounterfactualEvaluater(
            pattern=tuple(),
            y_pred=self.prediction,
            features=transformed_features,
            preds=preds,
            classes=classes,
        )

    def hill_climb(self):
        self.rule_miner.hill_climb()

    @property
    def best_pattern(self):
        return self.rule_miner.best_pattern

    @property
    def best_stability(self):
        return self.rule_miner.best_stability

    @property
    def best_excl_cov(self):
        return self.rule_miner.best_excl_cov

    @property
    def best_entropy(self):
        return self.rule_miner.best_entropy
    
    @property
    def best_precision(self):
        return self.rule_miner.best_precision
    
    @property
    def best_coverage(self):
        return self.rule_miner.best_coverage
    
    @cached_property
    def counterfactual_evaluator(self):
        if self.best_pattern:
            self._counterfactual_evaluator.pattern = self.best_pattern
        return self._counterfactual_evaluator
