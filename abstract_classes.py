from abc import ABC, abstractmethod


class Model(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def load(self, file_path):
        pass

    @abstractmethod
    def save(self, file_path):
        pass