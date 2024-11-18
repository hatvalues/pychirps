from data_preprocs.data_loading import DataProvider
from typing import Optional, Any

class InstanceEncoder():
    def __init__(self, provider: DataProvider, given_instance: Optional[dict[str, Any]] = None) -> None:
        self.column_descriptors = provider.column_descriptors
        if not given_instance:
            self._given_instance = {key: None for key in self.column_descriptors}
        else:
            self._given_instance = {key: given_instance.get(key) for key in self.column_descriptors}

    @property
    def given_instance(self):
        return self._given_instance
    
    @given_instance.setter
    def given_instance(self, **kwargs):
        self._given_instance = self._given_instance | kwargs
