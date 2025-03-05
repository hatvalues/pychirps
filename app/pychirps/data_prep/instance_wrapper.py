from app.pychirps.data_prep.data_provider import DataProvider
from typing import Optional, Any
from enum import Enum


class ColumnType(Enum):
    CATEGORICAL = ("categorical", "bool")
    INTEGER = ("ordinal", "count")
    FLOAT = ("float",)
    CONSTANT = ("constant",)


class IntegerType(Enum):
    ORDINAL = "ordinal"
    COUNT = "count"


class InstanceWrapper:
    def __init__(
        self, provider: DataProvider, given_instance: Optional[dict[str, Any]] = None
    ) -> None:
        self.feature_descriptors = provider.column_descriptors
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
        # defensive screening of input values
        update_values = {
            k: v for k, v in update_values.items() if k in self.feature_descriptors
        }

        # current values + update values + constant values that aren't presented to the user because they are a dataset aberration
        instance = (
            self.given_instance
            | update_values
            | {
                k: v.unique_values[0]
                for k, v in self.feature_descriptors.items()
                if v.otype in ColumnType.CONSTANT.value
            }
        )

        # validate the given values
        print(instance)
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

    @staticmethod
    def valide_stub(column: str, value: Any):
        pass

    def validator(self, column: str):
        if self.feature_descriptors[column].otype in ColumnType.CATEGORICAL.value:
            return self.validate_categorical
        elif (
            self.feature_descriptors[column].otype
            in ColumnType.INTEGER.value + ColumnType.FLOAT.value
        ):
            return self.validate_numeric
        return self.valide_stub  # pass constant values
