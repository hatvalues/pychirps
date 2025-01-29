# Just for dev - map out how the app will work with hard-coded data and model
# In due time - it will be adaptable based on uploaded data, and we'd have a model repository
from data_preprocs.data_providers.cervical import cervicalb_pd
from app.pychirps.explain.pre_explanations import predict
from ui.explanation_page import build_page_objects, create_sidebar
from app.pychirps.path_mining.classification_trees import random_forest_paths_factory
from app.pychirps.path_mining.forest_explorer import ForestExplorer
from app.pychirps.rule_mining.pattern_miner import PatternMiner
from app.pychirps.rule_mining.rule_miner import RuleMiner
import pandas as pd
import numpy as np
import streamlit as st


encoder, model, instance_wrapper = build_page_objects(cervicalb_pd)
transformed_features, transformed_targets = encoder.transform()

form_submit, input_values = create_sidebar(instance_wrapper.column_descriptors)


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
        encoder=encoder
    )
    st.markdown("### Model Predicts:")
    st.markdown(f"CLASS LABEL: {encoder.label_encoder.inverse_transform(model_prediction)[0]}")
    st.markdown(f"encoded value: {model_prediction[0]}")

    forest_explorer = ForestExplorer(model, encoder)
    instance32 = feature_frame.to_numpy().astype(np.float32).reshape(1, -1)
    forest_path = random_forest_paths_factory(forest_explorer, instance32)
    
    pattern_miner = PatternMiner(
        forest_path=forest_path,
        feature_names=encoder.preprocessor.get_feature_names_out().tolist(),
        prediction=model_prediction[0],
        min_support=min_support,
    )

    rule_miner = RuleMiner(
        pattern_miner=pattern_miner,
        y_pred=model_prediction[0],
        features=transformed_features,
        preds=model.predict(encoder.features),
        classes=np.unique(transformed_targets),
    )
    rule_miner.hill_climb()


    st.markdown("### Explanation:")
    st.json(rule_miner.best_pattern)
    st.markdown(f"Entropy: {rule_miner.entropy_score(rule_miner.best_pattern)}")
    st.markdown(f"Stability: {rule_miner.best_stability}")
    st.markdown(f"Exclusive Coverage: {rule_miner.best_excl_cov}")
    