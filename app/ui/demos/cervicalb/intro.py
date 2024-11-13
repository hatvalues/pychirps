# Just for dev - map out how the app will work with hard-coded data
# In due time - it will be adaptable based on uploaded data and model
from data_preprocs.data_providers import cervicalb_pd as cvb
import pandas as pd
import streamlit as st

st.title("Cervical Demo")
st.write(cvb.spiel)

st.sidebar.metric(label="Rows:", value=cvb.features.shape[0])
st.sidebar.metric(label="Columns:", value=cvb.features.shape[1])

columns_df = pd.DataFrame(cvb.features.columns, columns=["Features:"])
st.sidebar.table(columns_df)