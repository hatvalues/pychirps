import streamlit as st

pg = st.navigation([st.Page("cervicalb/intro.py"), st.Page("cervicalb/chirps.py")])
pg.run()