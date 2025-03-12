# Just for dev - map out how the app will work with hard-coded data and model
# In due time - it will be adaptable based on uploaded data, and we'd have a model repository
from data_preprocs.data_providers.cervical import cervicalb_pd
from ui.explanation_page import (
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
import pandas as pd
import numpy as np
import streamlit as st


encoder, model, instance_wrapper = build_page_objects(cervicalb_pd)

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
        dummy_target_class=pd.Series(cervicalb_pd.positive_class),
        encoder=encoder,
    )

    page_post_pred_texts(encoder, model_prediction)

    st.markdown(f"### Explanation:")

    explainer = Explainer(
        model=model,
        encoder=encoder,
        instance=instance_wrapper.given_instance_frame.to_numpy()
        .astype(np.float32)
        .reshape(1, -1),
        prediction=model_prediction[0],
        min_support=min_support,
    )
    explainer.hill_climb()

    rule_parser = RuleParser(
        feature_names=encoder.preprocessor.get_feature_names_out().tolist(),
        feature_descriptors=instance_wrapper.feature_descriptors,
    )

    page_explain_texts(explainer)

    counterfactuals = explainer.counterfactual_evaluator

    evaluted_counterfactuals = counterfactuals.evaluate_counterfactuals()
    st.markdown("### Counterfactuals")
    print(evaluted_counterfactuals)

    page_rule_frame(explainer, rule_parser, counterfactuals)

    st.plotly_chart(plot_partition(explainer.best_coverage, explainer.best_precision))

    page_post_explain_texts(explainer)
