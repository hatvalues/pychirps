from app.pychirps.data_prep.pandas_encoder import PandasEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from dataclasses import dataclass
from typing import Union, Optional


@dataclass(frozen=True)
class TreeNode:
    feature: np.uint8
    feature_name: str
    value: float
    threshold: float
    leq_threshold: bool


@dataclass(frozen=True)
class TreePath:
    prediction: np.uint8
    nodes: tuple[TreeNode]
    weight: float


@dataclass(frozen=True)
class ForestPath:
    paths: tuple[TreePath]

    def get_paths_for_prediction(
        self, prediction: Optional[np.uint8] = None
    ) -> tuple[tuple[TreeNode], float]:
        return tuple(
            (path.nodes, path.weight)
            for path in self.paths
            if path.prediction == prediction
        )


class ForestExplorer:
    def __init__(
        self,
        model: Union[RandomForestClassifier, AdaBoostClassifier],
        encoder: PandasEncoder,
    ) -> None:
        self.model = model
        self.trees = model.estimators_
        # AdaBoost SAMME has individual tree weights
        if not hasattr(model, "estimator_weights_"):
            self.tree_weights = np.ones(len(model.estimators_))
        else:
            self.tree_weights = model.estimator_weights_

        self.feature_names = encoder.preprocessor.get_feature_names_out().tolist()
        self.enumerated_feature_names = {i: v for i, v in enumerate(self.feature_names)}

    def parse_tree_for_instance(
        self,
        tree: DecisionTreeClassifier,
        instance: np.ndarray,
        path_weight: float,
    ) -> TreePath:
        prediction = tree.predict(instance)[0]
        features = tree.tree_.feature
        thresholds = tree.tree_.threshold
        sparse_path = tree.tree_.decision_path(instance).indices.tolist()[
            :-1
        ]  # exclude the final leaf node
        return TreePath(
            prediction=prediction,
            nodes=tuple(
                TreeNode(
                    feature=features[node],
                    feature_name=self.enumerated_feature_names.get(features[node]),
                    value=instance[0, features[node]],
                    threshold=thresholds[node],
                    leq_threshold=instance[0, features[node]] <= thresholds[node],
                )
                for node in sparse_path
            ),
            weight=path_weight,
        )

    def get_forest_path(self, instance: np.ndarray) -> ForestPath:
        return ForestPath(
            paths=tuple(
                self.parse_tree_for_instance(tree, instance, weight)
                for tree, weight in zip(self.trees, self.tree_weights)
            ),
        )
