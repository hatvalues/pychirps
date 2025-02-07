from app.pychirps.data_prep.data_provider import DataProvider
from typing import Optional, Any
from enum import Enum


class ColumnType(Enum):
    CATEGORICAL = ("categorical", "bool", "constant")
    INTEGER = ("ordinal", "count")
    FLOAT = ("float",)


class IntegerType(Enum):
    ORDINAL = "ordinal"
    COUNT = "count"


class InstanceWrapper:
    def __init__(
        self, provider: DataProvider, given_instance: Optional[dict[str, Any]] = None
    ) -> None:
        self.feature_descriptors = provider.feature_descriptors
        if not given_instance:
            self._given_instance = {key: None for key in self.feature_descriptors}
        else:
            self._given_instance = {
                key: given_instance.get(key) for key in self.feature_descriptors
            }

    @property
    def given_instance(self):
        return self._given_instance

    @given_instance.setter
    def given_instance(self, update_values: dict[str, Any]):
        update_values = {
            k: v for k, v in update_values.items() if k in self.feature_descriptors
        }
        instance = self.given_instance | update_values
        [self.validator(column)(column, value) for column, value in instance.items()]
        self._given_instance = instance

    def validate_categorical(self, column: str, value: Any):
        assert value in self.feature_descriptors[column].unique_values

    def validate_numeric(self, column: str, value: Any):
        assert (
            self.feature_descriptors[column].min
            <= value
            <= self.feature_descriptors[column].max
        )

    def validator(self, column: str):
        if self.feature_descriptors[column].otype in ColumnType.CATEGORICAL.value:
            return self.validate_categorical
        return self.validate_numeric
