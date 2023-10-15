from abc import ABC, abstractmethod


class Model(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def train(self, ds_train, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, question):
        pass

    @abstractmethod
    def evaluate(self, ds_test, *args, **kwargs):
        pass

    @abstractmethod
    def load(self, file_path):
        pass

    @abstractmethod
    def save(self, file_path):
        pass