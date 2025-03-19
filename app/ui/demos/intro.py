import streamlit as st

st.title("Welcome to PyCHIRPS!")
st.markdown("PyCHIRPS is a Python library for generating explanations for black box machine learning models.")
st.markdown("## Overview")
st.markdown("CHIRPS, Ada-WHIPS, and gbt-HIPS are methods for explaining how a model classifies any input instance.")
st.markdown("Unlike other well-known explanation methods, CHIRPS, Ada-WHIPS, and gbt-HIPS are model-specific, meaning they are designed to work with tree ensemble models.")
st.markdown("""- *Pros*: The explanations are drawn from model internals, so are a much better representation of the model's decision-making process over the background distribution.")
- *Cons*: Only works with decision tree ensembles.""")
st.markdown("The specific models we support are Random Forest Classifiers, AdaBoost SAMME, AdaBoost SAMME.R, and Gradient Boosting Classifiers from scikit-learn.")
st.markdown("Contributions for XGBoost, CatBoost, and LightGBM are welcome!")
st.markdown("Features:")
st.markdown("""- *Rule Mining*: Extracts high coverage and precision rule(s) from a the black box model using as few features as possible for better interpretability
- *Counterfactuals*: Generates counterfactual explanations that logically negate each of the rule antecedents, such that the instance is no longer covered
- *Counterfactual Statistics*: Evaluates comparative statistics for each counterfactual, showing the relative importance of each feature/value used in the explanation""")
st.markdown("## Demos")
st.markdown("To get started understanding how it works, select a demo from the sidebar.")
st.markdown("These demos based on common ML use cases e.g. UCI Machine Learning Repository, so you can see how PyCHIRPS works with familiar datasets before using it on your own use cases.")
st.markdown("""- Read about the dataset on the Data Description page
- Navigate to the CHIRPS page to build a Random Forest model (or retrieve a pre-built model from cache)
- Input your own instance values on the CHIRPS page side panel
- Trigger the model to predict a class label for your inputs and show the explanation for by clicking
the submit button""")
st.markdown("TO DO:")
st.markdown("""- Add support for more tree ensemble methods
- Add support for more datasets
- Model cache
- Input cache
- Finish and document the APIs""")
