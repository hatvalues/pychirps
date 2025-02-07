# Just for dev - map out how the app will work with hard-coded data and model
# In due time - it will be adaptable based on uploaded data, and we'd have a model repository
from data_preprocs.data_providers.cervical import cervicalb_pd
from app.pychirps.explain.pre_explanations import predict
from ui.explanation_page import build_page_objects, create_sidebar
from app.pychirps.explain.explainer import Explainer
from app.pychirps.explain.explanations import RuleParser
import pandas as pd
import numpy as np
import streamlit as st


encoder, model, instance_wrapper = build_page_objects(cervicalb_pd)

form_submit, input_values = create_sidebar(instance_wrapper.feature_descriptors)


st.markdown(f"""### Your RF Model.
:violet[***Out Of Bag Error:*** {round(1 - model.oob_score_, 4)}]""")

st.markdown("""Use the side panel to configure inputs, then click submit.
            
*Note: numerical input ranges represent the in distribution (observed) values.
Setting this values to the min or max is equivalent to setting any lower or higher number respectively.*""")


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
    st.markdown("### Model Predicts:")
    st.markdown(
        f"CLASS LABEL: {encoder.label_encoder.inverse_transform(model_prediction)[0]}"
    )
    st.markdown(f"encoded value: {model_prediction[0]}")

    explainer = Explainer(
        model, encoder, feature_frame, model_prediction[0], min_support
    )
    explainer.hill_climb()

    rule_parser = RuleParser(
        feature_names=encoder.preprocessor.get_feature_names_out().tolist(),
        feature_descriptors=instance_wrapper.feature_descriptors,
    )
    rule = rule_parser.parse(explainer.best_pattern, y_pred=model_prediction[0], rounding=2)
    rule_frame = pd.DataFrame(rule, columns=["Terms"])

    st.markdown(f"### Explanation:")
    st.table(rule_frame)
    st.markdown(f"Entropy: {explainer.best_entropy}")
    st.markdown(f"Stability: {explainer.best_stability}")
    st.markdown(f"Exclusive Coverage: {explainer.best_excl_cov}")
