# Just for dev - map out how the app will work with hard-coded data and model
# In due time - it will be adaptable based on uploaded data, and we'd have a model repository
from data_preprocs.data_providers import cervicalb_pd as cvb
from app.pychirps.path_mining.classification_trees import random_forest_paths_factory, ForestPath, ForestExplorer
from app.pychirps.prepare_data.pandas_encoder import PandasEncoder
from sklearn.ensemble import RandomForestClassifier
from app.config import DEFAULT_RANDOM_SEED
import numpy as np
import streamlit as st

@st.cache_resource
def fetch_clean_data(reset: bool = False):
    encoder = PandasEncoder(cvb.features, cvb.target)
    encoder.fit()
    return encoder

@st.cache_data
def transform_data(_encoder: PandasEncoder, reset: bool = False):
    return _encoder.transform()

encoder = fetch_clean_data()
transformed_features, transformed_target = transform_data(_encoder=encoder)

@st.cache_resource
def fit_model(features: np.ndarray, target: np.ndarray, reset: bool = False, **kwargs) -> RandomForestClassifier:
    hyper_parameter_defaults = {
        "n_estimators": 1000,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "oob_score": True,
    } | kwargs
    model = RandomForestClassifier(random_state=DEFAULT_RANDOM_SEED, **hyper_parameter_defaults)
    model.fit(features, target)
    return model

model = fit_model(features=transformed_features, target=transformed_target)

st.sidebar.metric(label="Fitted RF Model OOB Score:", value=round(model.oob_score_, 4))

@st.cache_resource
def fit_forest_explorer(encoder: PandasEncoder, model: RandomForestClassifier) -> ForestPath:
    return ForestExplorer(model, encoder)
    
