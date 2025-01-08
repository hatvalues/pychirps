import pytest
import pandas as pd
from tests.fixture_helper import load_yaml_fixture_file
from data_preprocs.data_providers.cervical import cervicalb_pd
from app.pychirps.data_prep.pandas_encoder import get_fitted_encoder_pd
from app.pychirps.model_prep.model_building import fit_random_forest
from app.pychirps.data_prep.instance_wrapper import InstanceWrapper
from app.pychirps.explain.pre_explanations import predict

encoder = get_fitted_encoder_pd(cervicalb_pd)
transformed_features, transformed_target = encoder.transform()
model = fit_random_forest(X=transformed_features, y=transformed_target, n_estimators=100)
instance_wrapper = InstanceWrapper(cervicalb_pd)

input_1 = load_yaml_fixture_file("pre_exp_predict_input_1")
input_2 = load_yaml_fixture_file("pre_exp_predict_input_2")

@pytest.mark.parametrize(
    "input,expected",
    [
        (input_1, [0]),
        (input_2, [1])
    ])
def test_predict(input, expected):
    instance_wrapper.given_instance = input

    feature_frame = pd.DataFrame(
        {k: [v] for k, v in instance_wrapper.given_instance.items()}
    )

    model_prediction = predict(
        model=model,
        feature_frame=feature_frame,
        dummy_target_class=pd.Series(cervicalb_pd.positive_class),
        encoder=encoder
    )

    assert model_prediction == expected