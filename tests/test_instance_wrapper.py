from app.pychirps.data_prep.instance_wrapper import InstanceWrapper
from tests.fixture_helper import assert_dict_matches_fixture
from dataclasses import asdict
import pytest


def test_instance_wrapper_cervical(cervicalb_pd):
    instance_wrapper = InstanceWrapper(cervicalb_pd)
    assert_dict_matches_fixture(
        {k: asdict(v) for k, v in instance_wrapper.column_descriptors.items()},
        "feature_descriptors_cervicalb",
    )
    assert_dict_matches_fixture(
        instance_wrapper.given_instance, "no_given_instance_cervicalb"
    )


def test_instance_wrapper_nursery(nursery_pd):
    instance_wrapper = InstanceWrapper(nursery_pd)
    assert_dict_matches_fixture(
        {k: asdict(v) for k, v in instance_wrapper.column_descriptors.items()},
        "feature_descriptors_nursery",
    )
    assert_dict_matches_fixture(
        instance_wrapper.given_instance, "no_given_instance_nursery"
    )


def test_instance_cervical(cervicalb_pd):
    instance = {k: v[0] for k, v in cervicalb_pd.features[0:1].to_dict().items()}
    instance_wrapper = InstanceWrapper(cervicalb_pd, instance)
    assert instance_wrapper.given_instance == instance


def test_instance_nursery(nursery_pd):
    instance = {k: v[0] for k, v in nursery_pd.features[0:1].to_dict().items()}
    instance_wrapper = InstanceWrapper(nursery_pd, instance)
    assert instance_wrapper.given_instance == instance


def test_instance_update_cervical(cervicalb_pd):
    instance = {k: v[0] for k, v in cervicalb_pd.features[0:1].to_dict().items()}
    instance_wrapper = InstanceWrapper(cervicalb_pd, instance)
    update_values = {
        "Age": 19.0,
        "Number of sexual partners": 5.0,
        "First sexual intercourse": 15.0,
        "DUMMY": None,
    }
    instance_wrapper.given_instance = update_values
    del update_values["First sexual intercourse"]
    del update_values["DUMMY"]
    assert instance_wrapper.given_instance == instance | update_values


def test_instance_update_nursery(nursery_pd):
    instance = {k: v[0] for k, v in nursery_pd.features[0:1].to_dict().items()}
    instance_wrapper = InstanceWrapper(nursery_pd, instance)
    update_values = {"children": "2"}
    instance_wrapper.given_instance = update_values
    assert instance_wrapper.given_instance == instance | update_values
    with pytest.raises(AssertionError):
        instance_wrapper.given_instance = {"parents": "DUMMY"}
