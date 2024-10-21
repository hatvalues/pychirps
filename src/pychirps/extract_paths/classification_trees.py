from sklearn.tree import DecisionTreeClassifier
from typing import Any
from src.pychirps.extract_paths.forest_metadata import ForestExplorer
import numpy as np


def get_instance_tree_path(tree: DecisionTreeClassifier, feature_names: dict[str, str], instance: np.ndarray) -> list[dict[str, Any]]:
    prediction = tree.predict(instance)[0]
    features = tree.tree_.feature
    thresholds = tree.tree_.threshold
    sparse_path = tree.tree_.decision_path(instance).indices.tolist()[:-1] # exclude the final leaf node
    path: list[dict[str, Any]] = [{} for _ in range(len(sparse_path))]
    for level, node in enumerate(sparse_path):
        path[level].update({
            'feature': features[node],
            'feature_name': feature_names.get(features[node]),
            'value': instance[0, features[node]],
            'threshold': thresholds[node],
            'leq_threshold': instance[0, features[node]] <= thresholds[node],
        })
    return prediction, path

def get_random_forest_paths(forest_explorer: ForestExplorer, instance: np.ndarray) -> list[tuple[int, list[dict[str, Any]]]]:
    forest_prediction = forest_explorer.model.predict(instance)
    feature_names = {i: v for i, v in enumerate(forest_explorer.feature_names)}
    return forest_prediction, [get_instance_tree_path(tree, feature_names, instance) for tree in forest_explorer.trees]

