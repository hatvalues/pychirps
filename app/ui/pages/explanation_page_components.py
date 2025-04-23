from app.pychirps.data_prep.data_provider import DataProvider, ColumnDescriptor
from app.pychirps.rule_mining.rule_miner import CounterfactualEvaluater
from app.pychirps.data_prep.pandas_encoder import get_fitted_encoder_pd, PandasEncoder
from app.pychirps.data_prep.instance_wrapper import InstanceWrapper, ColumnType
from app.pychirps.explain.explainer import Explainer
from app.pychirps.explain.explanations import RuleParser
from app.pychirps.model_prep.model_building import (
    fit_random_forest,
    RandomForestClassifier,
    fit_adaboost,
    AdaBoostClassifier,
)
from typing import Union, Any
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


@st.cache_resource
def fetch_fitted_encoder(
    data_provider: DataProvider,
    current_page: str,  # unique page name to reset the cache if a new demo is loaded
) -> PandasEncoder:
    return get_fitted_encoder_pd(data_provider)


@st.cache_data
def transform_data(
    _encoder: PandasEncoder, current_page: str
) -> tuple[np.ndarray, np.ndarray]:
    return _encoder.transform()


model_mapping = {
    "random_forest": fit_random_forest,
    "adaboost": fit_adaboost,
}


@st.cache_resource
def fit_model(
    model: str, features: np.ndarray, target: np.ndarray, current_page: str, **kwargs
) -> Union[RandomForestClassifier, AdaBoostClassifier]:
    return (
        model_mapping[model](X=features, y=target, random_state=42, **kwargs)
        if model in model_mapping
        else None
    )


@st.cache_resource
def fit_instance_wrapper(
    data_provider: DataProvider, current_page: str
) -> InstanceWrapper:
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
) -> tuple[bool, dict[ColumnDescriptor, Any], dict[str, Union[int, float]]]:
    with st.sidebar.form(key="input_form", border=False):
        st.title("Inputs")

        with st.expander("Configuration Options"):
            config_values = {
                "Frequent Pattern Support": st.number_input(
                    "Frequent Pattern Support (for finding rules)",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.1,
                )
            }

        with st.expander("Instance Values"):
            input_values = {
                column_name: render_input(column_name, column_descriptor)
                for column_name, column_descriptor in feature_descriptors.items()
            }
            form_submit = st.form_submit_button(label="Submit")
        return form_submit, input_values, config_values


def build_page_objects(
    data_provider: DataProvider, model: str, current_page: str
) -> tuple[
    PandasEncoder, Union[RandomForestClassifier, AdaBoostClassifier], InstanceWrapper
]:
    encoder = fetch_fitted_encoder(data_provider, current_page=current_page)
    transformed_features, transformed_target = transform_data(
        _encoder=encoder, current_page=current_page
    )
    model = fit_model(
        model=model,
        features=transformed_features,
        target=transformed_target,
        n_estimators=1000,
        current_page=current_page,
    )

    instance_wrapper = fit_instance_wrapper(data_provider, current_page=current_page)

    return encoder, model, instance_wrapper


def page_pre_submit_texts(model: Union[RandomForestClassifier, AdaBoostClassifier]):
    if type(model) == RandomForestClassifier:
        model_short_name = "RF"
        error_type = "Out Of Bag"
        error_value = 1 - model.oob_score_
    elif type(model) == AdaBoostClassifier:
        model_short_name = "AB"
        error_type = "Cross Validation"
        error_value = 1 - model.mean_cv_score
    else:
        model_short_name = "Unknown"
        error_type = "Unknown"
        error_value = 0.0
    st.markdown(
        f"""### Your {model_short_name} Model.
:violet[***{error_type} Error:*** {round(error_value, 4)}]"""
    )
    st.markdown(
        """Use the side panel to configure inputs, then click submit.
                
*Note: numerical input ranges represent the in distribution (observed) values.
Setting this values to the min or max is equivalent to setting any lower or higher number respectively.*"""
    )


def page_post_pred_texts(encoder: PandasEncoder, model_prediction: np.ndarray):
    st.markdown("### Model Predicts:")
    st.json(
        {
            "CLASS LABEL": encoder.label_encoder.inverse_transform(model_prediction)[0],
            "encoded value": int(model_prediction[0]),
        },
        expanded=False,
    )


def page_rule_frame(
    explainer: Explainer,
    rule_parser: RuleParser,
    counterfactual_evaluator: CounterfactualEvaluater,
):
    counterfactual_rules = counterfactual_evaluator.get_counterfactuals()
    evaluated_counterfactuals = np.array(
        counterfactual_evaluator.evaluate_counterfactuals()
    )

    counterfactual_precision = np.round(evaluated_counterfactuals[:, 1], 4)
    counterfactual_coverage = np.round(evaluated_counterfactuals[:, 0], 4)
    lost_precision = (
        f"{lp}%"
        for lp in np.round(
            (explainer.best_precision - counterfactual_precision)
            / explainer.best_precision
            * 100,
            2,
        )
    )
    lost_coverage = (
        f"{lc}%"
        for lc in np.round(
            (explainer.best_coverage - counterfactual_coverage)
            / explainer.best_coverage
            * 100
        )
    )
    rule_frame = pd.DataFrame(
        {
            "Counterfactual Rule": (
                " & ".join(
                    rule_parser.parse(
                        (node_pattern for node_pattern in counterfactual_rule),
                        rounding=2,
                    )
                )
                for counterfactual_rule in counterfactual_rules
            ),
            "Precision": counterfactual_precision,
            "Lost Precision %": lost_precision,
            "Coverage": counterfactual_coverage,
            "Lost Coverage %": lost_coverage,
        },
        dtype=str,
    )
    st.markdown("#### Rule and Counterfactuals:")
    st.table(rule_frame)


def page_explain_texts(
    explainer: Explainer,
    rule_parser: RuleParser,
    encoder: PandasEncoder,
    model_prediction: np.ndarray,
):
    st.markdown(
        "Your input is covered by the following rule:\n\n"
        + " *and*\n\n".join(rule_parser.parse(explainer.best_pattern, rounding=2))
        + "\n\n --> CLASS LABEL = "
        + encoder.label_encoder.inverse_transform(model_prediction)[0]
    )
    st.markdown("#### Rule Metrics:")
    st.markdown(
        f"Coverage: {round(explainer.best_coverage * 100, 2)}% of background distribution is covered by this rule's antecedent."
    )
    st.markdown(
        f"Precision: {round(explainer.best_precision * 100,2)}% of covered region has the same predicted class."
    )


def page_post_explain_texts(explainer: Explainer):
    st.markdown("#### Rule Finding Metrics:")
    st.markdown(f"Entropy: {explainer.best_entropy}")
    st.markdown(f"Stability: {explainer.best_stability}")
    st.markdown(f"Exclusive Coverage: {explainer.best_excl_cov}")


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
        # calculate x and y positions for the arrow tip / annotations - sometimes there is no area
        annotation_x: float = (1 - p) / 2 + 2 / max(
            len(region["label"]) for region in regions
        )
        annotation_y: float = (region["y0"] + region["y1"]) / 2
        if annotation_y > 0.95:
            ay = annotation_y - 0.05
        elif annotation_y < 0.05:
            ay = annotation_y + 0.05
        else:
            ay = annotation_y

        # check if there is an area to draw
        has_area = region["y0"] != region["y1"]

        # add the shape to the figure
        if has_area:
            fig.add_shape(
                type="rect",
                x0=region["x0"],
                x1=region["x1"],
                y0=region["y0"],
                y1=region["y1"],
                fillcolor=region["color"],
                line=dict(color="#000000", width=0.5),
            )

        # add annotation without an arrow if the region has zero area
        annotation_kwargs = {
            "x": p
            if has_area
            else annotation_x,  # Keep x consistent for the annotation itself
            "y": annotation_y if has_area else ay,
            "text": region["label"],
            "showarrow": has_area,
            "ax": annotation_x,  # Apply annotation_x regardless of has_area
            "ay": ay,
            "axref": "x",
            "ayref": "y",
            "font": dict(size=12, color="black"),
            "align": "left",
        }

        fig.add_annotation(**annotation_kwargs)

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

    st.plotly_chart(fig)
