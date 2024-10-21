from sklearn.tree import DecisionTreeClassifier
from typing import Any
from dataclasses import dataclass
from src.pychirps.extract_paths.forest_metadata import ForestExplorer
import numpy as np


@dataclass
class TreeNode:
    feature: int
    feature_name: str
    value: float
    threshold: float
    leq_threshold: bool


@dataclass
class TreePath:
    prediction: int
    nodes: list[TreeNode]


@dataclass
class ForestPath:
    prediction: int
    paths: list[TreePath]


def get_instance_tree_path(
    tree: DecisionTreeClassifier,
    feature_names: dict[str, str],
    instance: np.ndarray,
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
            )
            for node in sparse_path
        ],
    )


def get_random_forest_paths(
    forest_explorer: ForestExplorer,
    instance: np.ndarray,
) -> ForestPath:
    feature_names = {i: v for i, v in enumerate(forest_explorer.feature_names)}
    return ForestPath(
        prediction=forest_explorer.model.predict(instance)[0],
        paths=[
            get_instance_tree_path(tree, feature_names, instance)
            for tree in forest_explorer.trees
        ],
    )
