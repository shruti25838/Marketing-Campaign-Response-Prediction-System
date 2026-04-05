from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(self, **model_params):
        self.model_params = model_params

    @abstractmethod
    def build(self):
        raise NotImplementedError

    def name(self) -> str:
        return self.__class__.__name__
