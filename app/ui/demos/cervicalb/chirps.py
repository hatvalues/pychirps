# Just for dev - map out how the app will work with hard-coded data and model
# In due time - it will be adaptable based on uploaded data, and we'd have a model repository
from data_preprocs.data_providers import cervicalb_pd
from ui.explanation_page import (
    fetch_fitted_encoder,
    transform_data,
    fit_model,
    fit_instance_encoder,
    create_sidebar,
)


encoder = fetch_fitted_encoder(cervicalb_pd)
transformed_features, transformed_target = transform_data(_encoder=encoder)
model = fit_model(
    features=transformed_features, target=transformed_target, n_estimators=1000
)
instance_encoder = fit_instance_encoder(cervicalb_pd)

create_sidebar(model.oob_score_, instance_encoder.column_descriptors)
