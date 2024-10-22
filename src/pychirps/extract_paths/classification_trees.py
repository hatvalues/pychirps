from sklearn.tree import DecisionTreeClassifier
from dataclasses import dataclass
from src.pychirps.extract_paths.forest_metadata import ForestExplorer
from collections import defaultdict
import numpy as np


@dataclass
class TreeNode:
    feature: int
    feature_name: str
    value: float
    threshold: float
    leq_threshold: bool
    path_weight: float = 1.0


@dataclass
class TreePath:
    prediction: int
    nodes: list[TreeNode]


@dataclass
class GatheredTreePath:
    prediction: int
    paths: list[list[TreeNode]]


@dataclass
class ForestPath:
    prediction: int
    gathered_paths: list[GatheredTreePath]


def get_instance_tree_path(
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
        nodes=[
            TreeNode(
                feature=features[node],
                feature_name=feature_names.get(features[node]),
                value=instance[0, features[node]],
                threshold=thresholds[node],
                leq_threshold=instance[0, features[node]] <= thresholds[node],
                path_weight=path_weight,
            )
            for node in sparse_path
        ],
    )


def gather_tree_paths(paths=list[TreePath]) -> list[GatheredTreePath]:
    tree_paths_by_prediction = defaultdict(list)
    for path in paths:
        tree_paths_by_prediction[path.prediction].append(path.nodes)
    return [
        GatheredTreePath(
            prediction=prediction,
            paths=tree_paths_by_prediction[prediction],
        )
        for prediction in tree_paths_by_prediction
    ]


def get_random_forest_paths(
    forest_explorer: ForestExplorer,
    instance: np.ndarray,
) -> ForestPath:
    feature_names = {i: v for i, v in enumerate(forest_explorer.feature_names)}
    paths = [
        get_instance_tree_path(tree, feature_names, instance)
        for tree in forest_explorer.trees
    ]
    return ForestPath(
        prediction=forest_explorer.model.predict(instance)[0],
        gathered_paths=gather_tree_paths(paths),
    )
