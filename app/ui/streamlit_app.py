import streamlit as st
from app.ui.pages.data_description_factory import DataDescriptionPageFactory
from app.ui.pages.chirps_page_factory import ChirpsPageFactory
from data_preprocs.data_providers.cervical import cervicalb_pd
from data_preprocs.data_providers.nursery import nursery_pd

# Create Page Callables
cervicalb_data_description = DataDescriptionPageFactory(
    cervicalb_pd, "Cervical B Dataset"
).create_page()
cervicalb_chirps = ChirpsPageFactory(cervicalb_pd, "Cervical B Dataset").create_page()
nursery_data_description = DataDescriptionPageFactory(
    nursery_pd, "Nursery Dataset"
).create_page()
nursery_chirps = ChirpsPageFactory(nursery_pd, "Nursery Dataset").create_page()

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
