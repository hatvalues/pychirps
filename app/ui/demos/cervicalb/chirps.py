# Just for dev - map out how the app will work with hard-coded data and model
# In due time - it will be adaptable based on uploaded data, and we'd have a model repository
from data_preprocs.data_providers.cervical import cervicalb_pd
from app.pychirps.explain.pre_explanations import predict
from ui.explanation_page import (
    page_pre_submit_texts,
    page_post_pred_texts,
    page_rule_frame,
    page_post_explain_texts,
    build_page_objects,
    create_sidebar,
    plot_partition,
)
from app.pychirps.explain.explainer import Explainer
from app.pychirps.explain.explanations import RuleParser
import pandas as pd
import streamlit as st


encoder, model, instance_wrapper = build_page_objects(cervicalb_pd)

form_submit, input_values = create_sidebar(instance_wrapper.feature_descriptors)

page_pre_submit_texts(model)


if form_submit:
    min_support = input_values.pop("Frequent Pattern Support")
    instance_wrapper.given_instance = input_values
    st.markdown("### Your Inputs:")
    st.json(instance_wrapper.given_instance)
    feature_frame = pd.DataFrame(
        {k: [v] for k, v in instance_wrapper.given_instance.items()}
    )

    model_prediction = predict(
        model=model,
        feature_frame=feature_frame,
        dummy_target_class=pd.Series(cervicalb_pd.positive_class),
        encoder=encoder,
    )

    page_post_pred_texts(encoder, model_prediction)

    explainer = Explainer(
        model, encoder, feature_frame, model_prediction[0], min_support
    )
    explainer.hill_climb()

    rule_parser = RuleParser(
        feature_names=encoder.preprocessor.get_feature_names_out().tolist(),
        feature_descriptors=instance_wrapper.feature_descriptors,
    )

    counterfactuals = explainer.counterfactual_evaluator

    page_rule_frame(explainer, rule_parser, counterfactuals)

    page_post_explain_texts(explainer)

    st.plotly_chart(plot_partition(explainer.best_coverage, explainer.best_precision))

