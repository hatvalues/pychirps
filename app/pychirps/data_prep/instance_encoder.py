from app.pychirps.data_prep.data_provider import DataProvider
from typing import Optional, Any


class InstanceEncoder:
    def __init__(
        self, provider: DataProvider, given_instance: Optional[dict[str, Any]] = None
    ) -> None:
        self.column_descriptors = provider.column_descriptors
        if not given_instance:
            self._given_instance = {key: None for key in self.column_descriptors}
        else:
            self._given_instance = {
                key: given_instance.get(key) for key in self.column_descriptors
            }

    @property
    def given_instance(self):
        return self._given_instance

    @given_instance.setter
    def given_instance(self, update_values: dict[str, Any]):
        update_values = {
            k: v for k, v in update_values.items() if k in self.column_descriptors
        }
        instance = self.given_instance | update_values
        [self.validator(column)(column, value) for column, value in instance.items()]
        self._given_instance = instance

    def validate_categorical(self, column: str, value: Any):
        assert value in self.column_descriptors[column].unique_values

    def validate_numeric(self, column: str, value: Any):
        assert (
            self.column_descriptors[column].min
            <= value
            <= self.column_descriptors[column].max
        )

    def validator(self, column: str):
        if self.column_descriptors[column].otype == "categorical":
            return self.validate_categorical
        return self.validate_numeric
