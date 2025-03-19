# Description: This file contains the data_description_factory function which returns a function that displays the data description of a given data provider.
# The data_description_factory function takes in a title and a data_provider and returns a function that displays the data description of the given data provider.
# The returned function displays the title, spiel, number of rows, number of columns, and the features of the given data provider.
import pandas as pd
import streamlit as st
from typing import Callable
from app.pychirps.data_prep.data_provider import DataProvider


@st.cache_resource
def data_description_factory(
    title: str, data_provider: DataProvider
) -> Callable[[], None]:
    def data_description_page():
        st.title(title)
        st.write(data_provider.spiel)

        st.sidebar.metric(label="Rows:", value=data_provider.features.shape[0])
        st.sidebar.metric(label="Columns:", value=data_provider.features.shape[1])

        columns_df = pd.DataFrame(data_provider.features.columns, columns=["Features:"])
        st.sidebar.table(columns_df)

    # Streamlit is introspecting the function name and requires unique names for each page
    data_description_page.__name__ = f"data_description_{data_provider.name}"

    return data_description_page
