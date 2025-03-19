import streamlit as st
from app.ui.data_description_factory import data_description_factory
from app.ui.chirps_page_factory import chirps_page_factory
from data_preprocs.data_providers.cervical import cervicalb_pd
from data_preprocs.data_providers.nursery import nursery_pd

# Create Page Callables
cervicalb_data_description = data_description_factory(
    "Cervical B Dataset", cervicalb_pd
)
cervicalb_chirps = chirps_page_factory(cervicalb_pd)
nursery_data_description = data_description_factory("Nursery Dataset", nursery_pd)
nursery_chirps = chirps_page_factory(nursery_pd)

pg = st.navigation(
    {
        "Home": [st.Page("intro.py", title="Introduction")],
        "Cervical B Demo": [
            st.Page(cervicalb_data_description, title="Cervical B Dataset"),
            st.Page(cervicalb_chirps, title="CHIRPS (Random Forest)"),
        ],
        "Nursery Demo": [
            st.Page(nursery_data_description, title="Nursery Dataset"),
            st.Page(nursery_chirps, title="CHIRPS (Random Forest)"),
        ],
    }
)

pg.run()
