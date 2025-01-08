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

# input_values_1 = {
#     "Age":22,
#     "Number of sexual partners":7,
#     "First sexual intercourse":18,
#     "Num of pregnancies":6,
#     "Smokes":1,
#     "Smokes (years)":5,
#     "Smokes (packs/year)":5,
#     "Hormonal Contraceptives":1,
#     "Hormonal Contraceptives (years)":14,
#     "IUD":1,
#     "IUD (years)":1,
#     "STDs":1,
#     "STDs (number)":2,
#     "STDs:condylomatosis":1,
#     "STDs:cervical condylomatosis":0,
#     "STDs:vaginal condylomatosis":1,
#     "STDs:vulvo-perineal condylomatosis":1,
#     "STDs:syphilis":1,
#     "STDs:pelvic inflammatory disease":1,
#     "STDs:genital herpes":1,
#     "STDs:molluscum contagiosum":1,
#     "STDs:AIDS":0,
#     "STDs:HIV":0,
#     "STDs:Hepatitis B":0,
#     "STDs:HPV":0,
#     "STDs: Number of diagnosis":0,
#     "Dx:Cancer":0,
#     "Dx:CIN":0,
#     "Dx:HPV":0,
#     "Dx":0,
# }

# input_values_2 = {
#     "Age":35,
#     "Number of sexual partners":7,
#     "First sexual intercourse":14,
#     "Num of pregnancies":4,
#     "Smokes":1,
#     "Smokes (years)":15,
#     "Smokes (packs/year)":15,
#     "Hormonal Contraceptives":1,
#     "Hormonal Contraceptives (years)":14,
#     "IUD":1,
#     "IUD (years)":14,
#     "STDs":1,
#     "STDs (number)":4,
#     "STDs:condylomatosis":1,
#     "STDs:cervical condylomatosis":0,
#     "STDs:vaginal condylomatosis":1,
#     "STDs:vulvo-perineal condylomatosis":1,
#     "STDs:syphilis":1,
#     "STDs:pelvic inflammatory disease":1,
#     "STDs:genital herpes":1,
#     "STDs:molluscum contagiosum":1,
#     "STDs:AIDS":0,
#     "STDs:HIV":1,
#     "STDs:Hepatitis B":1,
#     "STDs:HPV":1,
#     "STDs: Number of diagnosis":3,
#     "Dx:Cancer":1,
#     "Dx:CIN":1,
#     "Dx:HPV":1,
#     "Dx":1,
# }

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