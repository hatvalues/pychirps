from abc import ABC, abstractmethod
from typing import Callable, Any
from app.pychirps.data_prep.data_provider import DataProvider

class PageFactory(ABC):
    def __init__(self, data_provider: Any, title: str):
        self.data_provider = data_provider
        self.title = title

    @abstractmethod
    def create_page(self) -> Callable[[], None]:
        pass
