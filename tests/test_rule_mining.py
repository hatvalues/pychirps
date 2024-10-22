from src.pychirps.extract_paths.classification_trees import get_random_forest_paths
from src.pychirps.extract_paths.forest_metadata import ForestExplorer
from sklearn.ensemble import RandomForestClassifier
from src.pychirps.pandas_utils.data_encoding import PandasEncoder
import data_preprocs.data_providers as dp
import numpy as np
from dataclasses import asdict
from tests.fixture_helper import assert_dict_matches_fixture, convert_native


def test_get_random_forest_paths():
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    encoder = PandasEncoder(dp.cervicalb_pd.features, dp.cervicalb_pd.target)
    encoder.fit()
    transformed_features, transformed_target = encoder.transform()
    model.fit(transformed_features, transformed_target)
    forest_explorer = ForestExplorer(model, encoder)

    instance = dp.cervicalh_pd.features.iloc[0]
    instance32 = instance.to_numpy().astype(np.float32).reshape(1, -1)
    forest_paths = get_random_forest_paths(forest_explorer, instance32)
