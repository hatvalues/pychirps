from app.pychirps.path_mining.forest_explorer import (
    ForestPath,
    TreePath,
    TreeNode,
)
import numpy as np
import pytest


@pytest.fixture
def weighted_paths():
    tree_node_1 = TreeNode(
        feature=np.int64(0),
        feature_name="num__Dx:CIN",
        value=np.float32(0.0),
        threshold=np.float64(0.5),
        leq_threshold=np.True_,
    )
    tree_node_2 = TreeNode(
        feature=np.int64(1),
        feature_name="num__STDs",
        value=np.float32(0.0),
        threshold=np.float64(0.5),
        leq_threshold=np.True_,
    )
    tree_node_3 = TreeNode(
        feature=np.int64(2),
        feature_name="num__Number of sexual partners",
        value=np.float32(4.0),
        threshold=np.float64(1.5),
        leq_threshold=np.False_,
    )
    tree_path_1 = TreePath(
        prediction=0, nodes=(tree_node_1, tree_node_2, tree_node_3), weight=0.23
    )
    tree_path_2 = TreePath(
        prediction=1, nodes=(tree_node_2, tree_node_3, tree_node_1), weight=0.13
    )
    tree_path_3 = TreePath(
        prediction=0, nodes=(tree_node_3, tree_node_1, tree_node_2), weight=0.35
    )
    return ForestPath(paths=(tree_path_1, tree_path_2, tree_path_3))
