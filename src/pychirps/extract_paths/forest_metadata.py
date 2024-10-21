from src.pychirps.pandas_utils.data_encoding import PandasEncoder
from sklearn.ensemble import RandomForestClassifier
import numpy as np


class ForestExplorer:
    def __init__(self, model: RandomForestClassifier, encoder: PandasEncoder) -> None:
        self.model = model
        self.encoder = encoder
        self.trees = model.estimators_
        if not hasattr(model, "estimator_weights_"):
            self.tree_weights = np.ones(len(model.estimators_))
        else:
            self.tree_weights = model.estimator_weights_

        self.feature_names = encoder.preprocessor.get_feature_names_out().tolist()
