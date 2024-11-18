# Just for dev - map out how the app will work with hard-coded data and model
# In due time - it will be adaptable based on uploaded data, and we'd have a model repository
from data_preprocs.data_providers import cervicalb_pd
from app.pychirps.path_mining.classification_trees import ForestPath, ForestExplorer
from app.pychirps.data_prep.pandas_encoder import get_fitted_encoder_pd, PandasEncoder
from app.pychirps.model_prep.model_building import fit_random_forest, RandomForestClassifier
import numpy as np
import streamlit as st

@st.cache_resource
def fetch_clean_data(reset: bool = False) -> PandasEncoder:
    return get_fitted_encoder_pd(cervicalb_pd)

@st.cache_data
def transform_data(_encoder: PandasEncoder, reset: bool = False) -> tuple[np.ndarray, np.ndarray]:
    return _encoder.transform()

encoder = fetch_clean_data()
transformed_features, transformed_target = transform_data(_encoder=encoder)

@st.cache_resource
def fit_model(features: np.ndarray, target: np.ndarray, reset: bool = False, **kwargs) -> RandomForestClassifier:
    return fit_random_forest(X=transformed_features, y=transformed_target, **kwargs)

model = fit_model(features=transformed_features, target=transformed_target, n_estimators=1000)

st.sidebar.metric(label="Fitted RF Model OOB Score:", value=round(model.oob_score_, 4))

@st.cache_resource
def fit_forest_explorer(encoder: PandasEncoder, model: RandomForestClassifier) -> ForestPath:
    return ForestExplorer(model, encoder)
    
