from app.pychirps.data_prep.instance_encoder import InstanceEncoder
from tests.fixture_helper import assert_dict_matches_fixture
from dataclasses import asdict

def test_instance_encoder_cervical(cervicalb_pd):
    instance_encoder = InstanceEncoder(cervicalb_pd)
    assert_dict_matches_fixture({k: asdict(v) for k, v in instance_encoder.column_descriptors.items()}, "column_descriptors_cervicalb")
    assert_dict_matches_fixture(instance_encoder.given_instance, "no_given_instance_cervicalb")

def test_instance_encoder_nursery(nursery_pd):
    instance_encoder = InstanceEncoder(nursery_pd)
    assert_dict_matches_fixture({k: asdict(v) for k, v in instance_encoder.column_descriptors.items()}, "column_descriptors_nursery")
    assert_dict_matches_fixture(instance_encoder.given_instance, "no_given_instance_nursery")