import os
import yaml
import numpy as np
from typing import Optional, Any, Union

fixture_path = os.path.realpath(
    f"{os.path.dirname(os.path.realpath(__file__))}/file_fixtures"
)
update_fixtures = os.environ.get("UPDATE_FILE_FIXTURES", False)


def convert_value_primitive(val):
    if isinstance(val, np.generic):
        return val.item()
    return val


def convert_native(data: Union[dict[str, Any], Any]) -> Any:
    # Convert each value in each dictionary in the list
    if isinstance(data, np.ndarray):
        data = data.tolist()
    if isinstance(data, list) or isinstance(data, tuple):
        return [convert_value_primitive(val) for val in data]
    elif isinstance(data, dict):
        return {key: convert_value_primitive(val) for key, val in data.items()}
    else:
        return convert_value_primitive(data)


def _fixture_file_path(fixture: str, file_extension: str = "yaml") -> str:
    return f"{fixture_path}/{fixture}.{file_extension}"


def load_yaml_fixture_file(fixture: str, if_not_exist: Optional[dict] = None) -> dict:
    fixture_file = _fixture_file_path(fixture)
    try:
        with open(fixture_file, "r", encoding="utf-8") as file:
            parsed_fixture = yaml.safe_load(file)
            if isinstance(parsed_fixture, dict):
                return parsed_fixture
            else:
                raise ValueError(f"Fixture {fixture} is not parsed to a dictionary")
    except FileNotFoundError as E:
        if if_not_exist is not None:
            return if_not_exist
        else:
            raise E


def assert_dict_matches_fixture(response: dict, fixture: str) -> None:
    response_file = _fixture_file_path(fixture)

    if update_fixtures:
        with open(response_file, "w", encoding="utf-8") as file:
            yaml.dump(response, file, allow_unicode=True)

    expected_response = load_yaml_fixture_file(fixture, if_not_exist={})

    assert response == expected_response


# def load_json_fixture_file(fixture, if_not_exist=None) -> dict:
#     fixture_file = _fixture_file_path(fixture, "json")
#     try:
#         with open(fixture_file, "r", encoding="utf-8") as file:
#             return json.load(file)
#     except FileNotFoundError as e:
#         if if_not_exist is not None:
#             return if_not_exist
#         else:
#             raise e


# def load_txt_fixture_file(fixture, if_not_exist=None) -> str:
#     fixture_file = _fixture_file_path(fixture, "txt")
#     try:
#         with open(fixture_file, "r", encoding="utf-8") as file:
#             return file.read()
#     except FileNotFoundError as e:
#         if if_not_exist is not None:
#             return if_not_exist
#         else:
#             raise e


# def assert_long_text_matches_fixture(response, fixture):
#     response_file = _fixture_file_path(fixture, "txt")
#     expected_response = load_txt_fixture_file(fixture, if_not_exist="")

#     if update_fixtures:
#         with open(response_file, "w", encoding="utf-8") as file:
#             file.write(response)

#     assert response.strip() == expected_response.strip()


# def assert_response_matches_fixture(response, fixture, body_ignore=None):
#     response_file = _fixture_file_path(fixture)
#     expected_response = load_yaml_fixture_file(fixture, if_not_exist={})

#     body = response.json()
#     if body_ignore:
#         for key in body_ignore:
#             del body[key]

#     if update_fixtures:
#         with open(response_file, "w", encoding="utf-8") as file:
#             updated_response = {
#                 "status_code": response.status_code,
#                 "body": body,
#             }
#             yaml.dump(updated_response, file, allow_unicode=True)

#     assert response.status_code == expected_response.get("status_code")
#     json_response = body
#     assert json_response == expected_response.get("body")
