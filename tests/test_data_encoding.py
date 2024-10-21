from pychirps.pandas_utils.data_encoding import PandasEncoder
from data_preprocs.data_providers import cervicalb_pd, nursery_pd
import numpy as np
import pandas as pd


def test_encode_pandas():
    encoder = PandasEncoder(cervicalb_pd.features, cervicalb_pd.target)
    encoder.fit()
    transformed_features, transformed_target = encoder.transform()
    assert transformed_features.shape == (858, 30)
    assert transformed_target.shape == (858,)
    assert isinstance(transformed_features, np.ndarray)
    assert isinstance(transformed_target, np.ndarray)
    assert encoder.label_encoder.classes_.tolist() == ["F", "T"]
    assert encoder.label_encoder.inverse_transform([0, 1]).tolist() == ["F", "T"]
    assert encoder.preprocessor.get_feature_names_out().tolist() == [
        "num__Age",
        "num__Number of sexual partners",
        "num__First sexual intercourse",
        "num__Num of pregnancies",
        "num__Smokes",
        "num__Smokes (years)",
        "num__Smokes (packs/year)",
        "num__Hormonal Contraceptives",
        "num__Hormonal Contraceptives (years)",
        "num__IUD",
        "num__IUD (years)",
        "num__STDs",
        "num__STDs (number)",
        "num__STDs:condylomatosis",
        "num__STDs:cervical condylomatosis",
        "num__STDs:vaginal condylomatosis",
        "num__STDs:vulvo-perineal condylomatosis",
        "num__STDs:syphilis",
        "num__STDs:pelvic inflammatory disease",
        "num__STDs:genital herpes",
        "num__STDs:molluscum contagiosum",
        "num__STDs:AIDS",
        "num__STDs:HIV",
        "num__STDs:Hepatitis B",
        "num__STDs:HPV",
        "num__STDs: Number of diagnosis",
        "num__Dx:Cancer",
        "num__Dx:CIN",
        "num__Dx:HPV",
        "num__Dx",
    ]

    encoder = PandasEncoder(nursery_pd.features, nursery_pd.target)
    encoder.fit()
    transformed_features, transformed_target = encoder.transform()
    assert transformed_features.shape == (12958, 27)
    assert transformed_target.shape == (12958,)
    assert isinstance(transformed_features, np.ndarray)
    assert isinstance(transformed_target, np.ndarray)
    assert encoder.preprocessor.get_feature_names_out().tolist() == [
        "cat__parents_great_pret",
        "cat__parents_pretentious",
        "cat__parents_usual",
        "cat__has_nurs_critical",
        "cat__has_nurs_improper",
        "cat__has_nurs_less_proper",
        "cat__has_nurs_proper",
        "cat__has_nurs_very_crit",
        "cat__form_complete",
        "cat__form_completed",
        "cat__form_foster",
        "cat__form_incomplete",
        "cat__children_1",
        "cat__children_2",
        "cat__children_3",
        "cat__children_more",
        "cat__housing_convenient",
        "cat__housing_critical",
        "cat__housing_less_conv",
        "cat__finance_convenient",
        "cat__finance_inconv",
        "cat__social_nonprob",
        "cat__social_problematic",
        "cat__social_slightly_prob",
        "cat__health_not_recom",
        "cat__health_priority",
        "cat__health_recommended",
    ]

    # Check that the transformed features match the original values
    row = nursery_pd.features.iloc[0]  # type: ignore # more mypy madness
    original_values = [f"{col}_{val}" for col, val in row.items()]
    # pick up just the transformed feature names where we get a 1/True
    transformed_values = [
        fname.replace("cat__", "")
        for fname, tfeat in zip(
            encoder.preprocessor.get_feature_names_out(), transformed_features[0]
        )
        if tfeat == 1
    ]
    assert original_values == transformed_values

    bool_example_features = pd.DataFrame(
        {
            "is_it": [True, False, True, False, True],
            "count_it": [1, 2, 3, 4, 5],
            "tell_it": ["yes", "no", "yes", "no", "maybe"],
        }
    )
    bool_example_target = pd.Series([True, False, True, False, True])
    encoder = PandasEncoder(bool_example_features, bool_example_target)
    encoder.fit()
    transformed_features, transformed_target = encoder.transform()
    assert transformed_features.shape == (5, 5)
    assert transformed_target.shape == (5,)
    assert isinstance(transformed_features, np.ndarray)
    assert isinstance(transformed_target, np.ndarray)
    assert encoder.preprocessor.get_feature_names_out().tolist() == [
        "num__is_it",
        "num__count_it",
        "cat__tell_it_maybe",
        "cat__tell_it_no",
        "cat__tell_it_yes",
    ]
