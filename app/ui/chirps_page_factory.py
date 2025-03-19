from app.ui.explanation_page_components import (
    page_pre_submit_texts,
    page_post_pred_texts,
    page_explain_texts,
    page_rule_frame,
    page_post_explain_texts,
    build_page_objects,
    create_sidebar,
    plot_partition,
)
from app.pychirps.explain.explainer import Explainer, predict
from app.pychirps.explain.explanations import RuleParser
from app.pychirps.data_prep.instance_wrapper import ColumnType
from app.pychirps.data_prep.data_provider import DataProvider
import pandas as pd
import numpy as np
import streamlit as st
from typing import Callable


def chirps_page_factory(data_provider: DataProvider) -> Callable[[], None]:
    def chirps_page():
        encoder, model, instance_wrapper = build_page_objects(data_provider)

        contants_check = {
            k: v.unique_values[0]
            for k, v in instance_wrapper.feature_descriptors.items()
            if v.otype in ColumnType.CONSTANT.value
        }

        form_submit, input_values = create_sidebar(instance_wrapper.feature_descriptors)

        page_pre_submit_texts(model)

        if form_submit:
            min_support = input_values.pop("Frequent Pattern Support")
            instance_wrapper.given_instance = input_values
            st.markdown("### Your Inputs:")
            st.json(instance_wrapper.given_instance, expanded=False)

            if contants_check:
                st.markdown("#### Note")
                st.markdown(
                    "The following dataset features exist in the data dictionary as constants:"
                )
                st.json(contants_check, expanded=False)

            model_prediction = predict(
                model=model,
                feature_frame=instance_wrapper.given_instance_frame,
                encoder=encoder,
            )

            page_post_pred_texts(encoder, model_prediction)

            st.markdown("### Explanation:")

            explainer = Explainer(
                model=model,
                encoder=encoder,
                feature_frame=instance_wrapper.given_instance_frame,
                prediction=model_prediction[0],
                min_support=min_support,
            )
            explainer.hill_climb()

            rule_parser = RuleParser(
                feature_names=encoder.preprocessor.get_feature_names_out().tolist(),
                feature_descriptors=instance_wrapper.feature_descriptors,
            )

            page_explain_texts(explainer, rule_parser, encoder, model_prediction)

            plot_partition(explainer.best_coverage, explainer.best_precision)

            counterfactual_evaluator = explainer.counterfactual_evaluator
            st.markdown("### Counterfactuals")
            page_rule_frame(explainer, rule_parser, counterfactual_evaluator)

            page_post_explain_texts(explainer)

    # Streamlit is introspecting the function name and requires unique names for each page
    chirps_page.__name__ = f"chirps_{data_provider.name}"

    return chirps_page
