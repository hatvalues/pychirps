# Just for dev - map out how the app will work with hard-coded data
# In due time - it will be adaptable based on uploaded data and model
from data_preprocs.data_providers import cervicalb_pd as cvb
import streamlit as st

# DEVELOPMENT VERSION - HARDCODED DEMO
st.title("Cervical Demo")
st.write(cvb.spiel)

st.sidebar.write(cvb.features.columns)

