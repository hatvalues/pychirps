from pychirps.path_mining.forest_explorer import ForestExplorer
from sklearn.ensemble import RandomForestClassifier
from app.pychirps.data_prep.pandas_encoder import PandasEncoder
from data_preprocs.data_providers.cervical import cervicalb_pd
import numpy as np


def test_forest_explorer():
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    encoder = PandasEncoder(cervicalb_pd.features, cervicalb_pd.target)
    encoder.fit()
    transformed_features, transformed_target = encoder.transform()
    model.fit(transformed_features, transformed_target)

    explorer = ForestExplorer(model, encoder)
    assert len(explorer.trees) == 10
    assert len(explorer.tree_weights) == 10
    assert isinstance(explorer.feature_names, list)
    assert isinstance(explorer.trees, list)
    assert isinstance(explorer.tree_weights, np.ndarray)
    assert explorer.tree_weights.all() == 1.0
    sparse_path = explorer.trees[0].tree_.decision_path(
        cervicalb_pd.features.loc[0].values.reshape(1, -1).astype(np.float32)
    )
    assert sparse_path.indices.tolist() == [
        0,
        1,
        2,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        64,
        65,
        75,
    ]
