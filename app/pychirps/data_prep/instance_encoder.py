from data_preprocs.data_loading import DataProvider

class InstanceEncoder():
    def __init__(self, provider: DataProvider) -> None:
        self.column_descriptors = provider.column_descriptors