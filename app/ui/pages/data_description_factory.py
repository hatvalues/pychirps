# Description: This file contains the data_description_factory function which returns a function that displays the data description of a given data provider.
# The data_description_factory function takes in a title and a data_provider and returns a function that displays the data description of the given data provider.
# The returned function displays the title, spiel, number of rows, number of columns, and the features of the given data provider.
import pandas as pd
import streamlit as st
from typing import Callable
from app.ui.pages.page_factory import PageFactory


class DataDescriptionPageFactory(PageFactory):
    def create_page(self) -> Callable[[], None]:
        def page():
            st.session_state["current_page_id"] = f"{self.title}_description"
            st.title(self.title)
            st.write(self.data_provider.spiel)

            st.sidebar.metric(label="Rows:", value=self.data_provider.features.shape[0])
            st.sidebar.metric(
                label="Columns:", value=self.data_provider.features.shape[1]
            )

            columns_df = pd.DataFrame(
                self.data_provider.features.columns, columns=["Features:"]
            )
            st.sidebar.table(columns_df)

        # Streamlit is introspecting the function name and requires unique names for each page
        page.__name__ = f"data_description_{self.data_provider.name}"

        return page
