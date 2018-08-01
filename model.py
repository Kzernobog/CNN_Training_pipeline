from abc import ABC, abstractmethod

class Model(ABC):
    """An abstract class that declares a template for the average model"""

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def initialize_weights(self):
        pass

    # @abstractmethod
    # def train(self):
    #     pass

    # @abstractmethod
    # def infer(self):
    #     pass

    @abstractmethod
    def compute_cost(self):
        pass

    @abstractmethod
    def model(self):
        pass

    @abstractmethod
    def forward_propagation(self):
        pass

if __name__ == "__main__":
    pass
