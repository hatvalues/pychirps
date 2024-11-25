from app.pychirps.data_prep.data_provider import DataProvider, ColumnDescriptor
from app.pychirps.path_mining.classification_trees import ForestPath, ForestExplorer
from app.pychirps.data_prep.pandas_encoder import get_fitted_encoder_pd, PandasEncoder
from app.pychirps.data_prep.instance_encoder import InstanceEncoder
from app.pychirps.model_prep.model_building import (
    fit_random_forest,
    RandomForestClassifier,
)
import numpy as np
import streamlit as st


@st.cache_resource
def fetch_fitted_encoder(
    data_provider: DataProvider, reset: bool = False
) -> PandasEncoder:
    return get_fitted_encoder_pd(data_provider)


@st.cache_data
def transform_data(
    _encoder: PandasEncoder, reset: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    return _encoder.transform()


@st.cache_resource
def fit_model(
    features: np.ndarray, target: np.ndarray, reset: bool = False, **kwargs
) -> RandomForestClassifier:
    return fit_random_forest(X=features, y=target, **kwargs)


@st.cache_resource
def fit_forest_explorer(
    encoder: PandasEncoder, model: RandomForestClassifier
) -> ForestPath:
    return ForestExplorer(model, encoder)


@st.cache_resource
def fit_instance_encoder(data_provider: DataProvider) -> InstanceEncoder:
    return InstanceEncoder(data_provider)


def create_sidebar(
    oob_score: float, column_descriptors: dict[ColumnDescriptor]
) -> None:
    with st.sidebar:
        oob_score = st.metric(
            label="Fitted RF Model OOB Score:", value=round(oob_score, 4)
        )
        for column_name, column_descriptor in column_descriptors.items():
            if column_descriptor.otype == "categorical":
                _ = st.radio(column_name, column_descriptor.unique_values)
            elif column_descriptor.otype in ("ordinal", "count"):
                _ = st.number_input(
                    column_name,
                    min_value=column_descriptor.min,
                    max_value=column_descriptor.max,
                    value="min"
                )
            else:
                st.slider(
                    column_name,
                    min_value=column_descriptor.min,
                    max_value=column_descriptor.max,
                )
