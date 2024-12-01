from app.pychirps.data_prep.data_provider import DataProvider, ColumnDescriptor
from app.pychirps.path_mining.classification_trees import ForestPath, ForestExplorer
from app.pychirps.data_prep.pandas_encoder import get_fitted_encoder_pd, PandasEncoder
from app.pychirps.data_prep.instance_wrapper import InstanceWrapper
from app.pychirps.model_prep.model_building import (
    fit_random_forest,
    RandomForestClassifier,
)
from typing import Union, Any
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
def fit_instance_wrapper(data_provider: DataProvider) -> InstanceWrapper:
    return InstanceWrapper(data_provider)


def render_categorical_input(
    column_name: str, column_descriptor: ColumnDescriptor
) -> st.radio:
    return st.radio(column_name, column_descriptor.unique_values)


def render_integer_input(
    column_name: str, column_descriptor: ColumnDescriptor
) -> st.number_input:
    return st.number_input(
        column_name,
        min_value=column_descriptor.min,
        max_value=column_descriptor.max,
        value="min",
    )


def render_float_input(
    column_name: str, column_descriptor: ColumnDescriptor
) -> st.slider:
    return st.slider(
        column_name,
        min_value=column_descriptor.min,
        max_value=column_descriptor.max,
    )


def render_input(column_name: str, column_descriptor: ColumnDescriptor) -> Any:
    if column_descriptor.otype in ("categorical", "bool", "const"):
        return render_categorical_input(column_name, column_descriptor)
    elif column_descriptor.otype in ("ordinal", "count"):
        return render_integer_input(column_name, column_descriptor)
    else:
        return render_float_input(column_name, column_descriptor)


def create_sidebar(
    column_descriptors: dict[ColumnDescriptor],
) -> dict[str, Union[int, float, str]]:
    with st.sidebar.form(key="input_form", border=False):
        input_values = {
            column_name: render_input(column_name, column_descriptor)
            for column_name, column_descriptor in column_descriptors.items()
        }
        form_submit = st.form_submit_button(label="Submit")
        return form_submit, input_values


def build_page_objects(
    data_provider: DataProvider,
) -> tuple[PandasEncoder, RandomForestClassifier, InstanceWrapper]:
    encoder = fetch_fitted_encoder(data_provider)
    transformed_features, transformed_target = transform_data(_encoder=encoder)
    model = fit_model(
        features=transformed_features, target=transformed_target, n_estimators=1000
    )

    instance_wrapper = fit_instance_wrapper(data_provider)

    return encoder, model, instance_wrapper
