from app.ui.demos import cervicalb
import streamlit as st

# DEVELOPMENT VERSION - HARDCODED DEMO
st.title("Cervical Demo")
st.write(cervicalb.spiel)

st.sidebar.write(cervicalb.column_names)

if st.button("Classify and Explain"):
    st.write(f"Prediction: 1")
    st.write("Explanation:", "rule ...")

