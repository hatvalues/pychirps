import streamlit as st

st.title("Welcome to PyChirps!")
st.markdown("PyChirps is a Python library for generating explanations for black box machine learning models.")
st.markdown("The specific models we support are Random Forest Classifiers, AdaBoost SAMME, AdaBoost SAMME.R, and Gradient Boosting Classifiers from scikit-learn.")
st.markdown("Contributions for XGBoost, CatBoost, and LightGBM are welcome!")
st.markdown("Features:")
st.markdown("""- *Rule Mining*: Extracts rules from a Random Forest model
- *Counterfactuals*: Generates counterfactual explanations for a given instance
- *Explanations*: Provides explanations for a given instance
- *Visualizations*: Visualizes the decision boundaries of a Random Forest model""")
st.markdown("To get started understanding how it works, select a demo from the sidebar.")
st.markdown("""- Read about the dataset on the Data Description page
- Input your own instance values on the CHIRPS page
- View the predicted class label and the explanation for the given class by clicking
the submit button""")
st.markdown("TO DO:")
st.markdown("""- Add support for more tree ensemble methods
- Add support for more datasets
- Finish and document the APIs""")
