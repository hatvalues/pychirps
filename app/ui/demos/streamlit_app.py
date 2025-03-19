import streamlit as st

introduction = st.Page("intro.py")
cervicalb_dd = st.Page("cervicalb/data_description.py", title="The Cervical B Dataset")
cervicalb_chirps = st.Page("cervicalb/chirps.py", title="CHIRPS (Random Forest)")

pg = st.navigation({
    "Home": [introduction],
    "Cervical B Demo": [cervicalb_dd, cervicalb_chirps,]
})

pg.run()