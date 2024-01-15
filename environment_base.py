import abc

class Environment(abc.ABC):

    @abc.abstractclassmethod
    def step():
        pass
    
    @property
    @abc.abstractmethod
    def population_size(self):
        pass

    @property
    @abc.abstractmethod
    def sigma(self):
        pass

