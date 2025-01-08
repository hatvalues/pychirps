import pandas as pd
import numpy as np
from app.pychirps.data_prep.pandas_encoder import PandasEncoder
from app.pychirps.model_prep.model_building import RandomForestClassifier


def predict(
        model: RandomForestClassifier,
        feature_frame: pd.DataFrame,
        dummy_target_class: str,
        encoder: PandasEncoder
    ) -> np.ndarray:
    dummy_target = pd.Series(dummy_target_class)
    encoded_instance, _ = encoder.transform(feature_frame, dummy_target)
    return model.predict(encoded_instance)
