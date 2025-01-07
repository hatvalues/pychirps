# Just for dev - map out how the app will work with hard-coded data and model
# In due time - it will be adaptable based on uploaded data, and we'd have a model repository
from data_preprocs.data_providers.cervical import cervicalb_pd
from ui.explanation_page import build_page_objects, create_sidebar
import pandas as pd
import streamlit as st


encoder, model, instance_wrapper = build_page_objects(cervicalb_pd)

form_submit, input_values = create_sidebar(instance_wrapper.column_descriptors)


st.markdown(f"""### Your RF Model.
:violet[***Out Of Bag Error:*** {round(1 - model.oob_score_, 4)}]""")

st.markdown("""Use the side panel to configure inputs, then click submit.
            
*Note: numerical input ranges represent the in distribution (observed) values.
Setting this values to the min or max is equivalent to setting any lower or higher number respectively.*""")


if form_submit:
    instance_wrapper.given_instance = input_values
    st.markdown("### Your Inputs:")
    st.json(instance_wrapper.given_instance)
    feature_frame = pd.DataFrame(
        {k: [v] for k, v in instance_wrapper.given_instance.items()}
    )
    dummy_target = pd.Series(cervicalb_pd.positive_class)
    encoded_instance, _ = encoder.transform(feature_frame, dummy_target)
    pred = model.predict(encoded_instance)
    st.dataframe(pred)
