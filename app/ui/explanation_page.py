from app.pychirps.data_prep.data_provider import DataProvider, ColumnDescriptor
from app.pychirps.path_mining.classification_trees import ForestPath, ForestExplorer
from app.pychirps.rule_mining.rule_miner import CounterfactualEvaluater
from app.pychirps.data_prep.pandas_encoder import get_fitted_encoder_pd, PandasEncoder
from app.pychirps.data_prep.instance_wrapper import InstanceWrapper, ColumnType
from app.pychirps.explain.explainer import Explainer
from app.pychirps.explain.explanations import RuleParser
from app.pychirps.model_prep.model_building import (
    fit_random_forest,
    RandomForestClassifier,
)
from typing import Union, Any
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


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


def binary_formatter(value: bool) -> str:
    return "Yes" if value else "No"


def render_binary_input(
    column_name: str, column_descriptor: ColumnDescriptor
) -> st.radio:
    return st.radio(
        column_name,
        sorted(column_descriptor.unique_values),
        format_func=binary_formatter,
        horizontal=True,
    )


def render_categorical_input(
    column_name: str, column_descriptor: ColumnDescriptor
) -> st.radio:
    return st.radio(column_name, column_descriptor.unique_values)


def render_integer_input(
    column_name: str, column_descriptor: ColumnDescriptor
) -> st.slider:
    return st.slider(
        column_name,
        min_value=int(column_descriptor.min),
        max_value=int(column_descriptor.max),
        step=1,
    )


def render_float_input(
    column_name: str, column_descriptor: ColumnDescriptor
) -> st.number_input:
    return st.number_input(
        column_name,
        min_value=column_descriptor.min,
        max_value=column_descriptor.max,
        value="min",
    )


def render_input(column_name: str, column_descriptor: ColumnDescriptor) -> Any:
    if column_descriptor.otype == "constant":
        pass
    elif column_descriptor.otype == "bool":
        return render_binary_input(column_name, column_descriptor)
    elif column_descriptor.otype in ColumnType.CATEGORICAL.value:
        return render_categorical_input(column_name, column_descriptor)
    elif column_descriptor.otype in ColumnType.INTEGER.value:
        return render_integer_input(column_name, column_descriptor)
    else:
        return render_float_input(column_name, column_descriptor)


def create_sidebar(
    feature_descriptors: dict[ColumnDescriptor],
) -> dict[str, Union[int, float, str]]:
    with st.sidebar.form(key="input_form", border=False):
        input_values = {
            "Frequent Pattern Support": st.number_input(
                "Frequent Pattern Support",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
            )
        } | {
            column_name: render_input(column_name, column_descriptor)
            for column_name, column_descriptor in feature_descriptors.items()
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


def page_pre_submit_texts(model: RandomForestClassifier):
    st.markdown(
        f"""### Your RF Model.
:violet[***Out Of Bag Error:*** {round(1 - model.oob_score_, 4)}]"""
    )

    st.markdown(
        """Use the side panel to configure inputs, then click submit.
                
*Note: numerical input ranges represent the in distribution (observed) values.
Setting this values to the min or max is equivalent to setting any lower or higher number respectively.*"""
    )


def page_post_pred_texts(encoder: PandasEncoder, model_prediction: np.ndarray):
    st.markdown("### Model Predicts:")
    st.markdown(
        f"CLASS LABEL: {encoder.label_encoder.inverse_transform(model_prediction)[0]}"
    )
    st.markdown(f"encoded value: {model_prediction[0]}")


def page_rule_frame(
    explainer: Explainer, rule_parser: RuleParser, counterfactual_evaluator: CounterfactualEvaluater
):
    counterfactuals = np.array(counterfactual_evaluator.evaluate_counterfactuals())
    # each row is [entropy, coverage, precision]
    counterfactual_precision = counterfactuals[:, 1]
    counterfactual_coverage = counterfactuals[:, 0]
    rule_frame = pd.DataFrame(
        {"Terms": rule_parser.parse(
            explainer.best_pattern, rounding=2
        ),
        "Contrast Precision": counterfactual_precision,
        "Contrasts (diff. Precision)": explainer.best_stability - counterfactual_precision,
        "Contrasts (rel. Precision)": (explainer.best_stability - counterfactual_precision) / explainer.best_stability,
        "Contrast Coverage": counterfactual_coverage,
        "Contrasts (diff. Coverage)": explainer.best_excl_cov - counterfactual_coverage,
        "Contrasts (rel. Coverage)": (explainer.best_excl_cov - counterfactual_coverage) / explainer.best_excl_cov,
        }
    )
    st.markdown(f"### Explanation:")
    st.table(rule_frame)


def page_post_explain_texts(explainer: Explainer):
    st.markdown(f"Entropy: {explainer.best_entropy}")
    st.markdown(f"Stability: {explainer.best_stability}")
    st.markdown(f"Exclusive Coverage: {explainer.best_excl_cov}")
    st.markdown(f"Coverage: {explainer.best_coverage}")
    st.markdown(f"Precision: {explainer.best_precision}")


def plot_partition(p: float, q: float):
    """Creates a partitioned visualization using Plotly."""
    fig = go.Figure(
        layout_xaxis_range=[0, 1],
        layout_yaxis_range=[0, 1],
    )

    # Define regions and their positions
    regions = [
        {
            "x0": 0,
            "x1": p,
            "y0": 1,
            "y1": q,
            "color": "#F7A399",
            "label": f"Other Class: {p * (1-q) * 100:.2f}%",
        },
        {
            "x0": 0,
            "x1": p,
            "y0": q,
            "y1": 0,
            "color": "#A9DEF9",
            "label": f"Same Class: {p * q * 100:.2f}%",
        },
    ]

    for region in regions:
        fig.add_shape(
            type="rect",
            x0=region["x0"],
            x1=region["x1"],
            y0=region["y0"],
            y1=region["y1"],
            fillcolor=region["color"],
            line=dict(color="#000000", width=0.5),
        )
        arrow_tip = (region["y0"] + region["y1"]) / 2
        if arrow_tip > 0.95:
            ay = arrow_tip - 0.05
        elif arrow_tip < 0.05:
            ay = arrow_tip + 0.05
        else:
            ay = arrow_tip
        fig.add_annotation(
            x=p,
            y=arrow_tip,
            text=region["label"],
            showarrow=True,
            arrowhead=2,
            ax=(1 - p) / 2 + 2 / len(region["label"]),
            ay=ay,
            axref="x",
            ayref="y",
            font=dict(size=12, color="black"),
            align="left",
        )

    # Layout adjustments
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        width=600,
        height=600,
        autosize=False,
        plot_bgcolor="white",
        margin=dict(l=0, r=0, t=0, b=0, pad=0),
    )

    return fig
