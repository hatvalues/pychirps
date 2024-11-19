from sklearn.tree import DecisionTreeClassifier
from dataclasses import dataclass
from app.pychirps.path_mining.forest_explorer import ForestExplorer
import numpy as np


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
    weight: float = 1.0


@dataclass(frozen=True)
class ForestPath:
    prediction: np.uint8
    paths: tuple[TreePath]

    def get_for_prediction(self, prediction: np.uint8) -> list[TreePath]:
        return tuple(
            (path.nodes, path.weight)
            for path in self.paths
            if path.prediction == prediction
        )


def instance_tree_factory(
    tree: DecisionTreeClassifier,
    feature_names: dict[str, str],
    instance: np.ndarray,
    path_weight: float = 1.0,
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
                feature_name=feature_names.get(features[node]),
                value=instance[0, features[node]],
                threshold=thresholds[node],
                leq_threshold=instance[0, features[node]] <= thresholds[node],
            )
            for node in sparse_path
        ),
        weight=path_weight,
    )


def random_forest_paths_factory(
    forest_explorer: ForestExplorer,
    instance: np.ndarray,
) -> ForestPath:
    feature_names = {i: v for i, v in enumerate(forest_explorer.feature_names)}
    return ForestPath(
        prediction=forest_explorer.model.predict(instance)[0],
        paths=tuple(
            instance_tree_factory(tree, feature_names, instance)
            for tree in forest_explorer.trees
        ),
    )
