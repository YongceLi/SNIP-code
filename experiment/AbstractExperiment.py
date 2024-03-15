from abc import ABC, abstractmethod

class AbstractExperiment(ABC):
    """
    Abstract Class for an experiment
    """

    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

    @abstractmethod
    def preprocess_data(self):
        """
        Process the dataset. Return a dataloader for the certain experiment.
        """
        pass

    @abstractmethod
    def run(self, dataloader):
        """
        evaluate the model on a certain task with the given dataloader.
        """
        pass
    
    @abstractmethod
    def postprocess(self, results):
        """
        postprocess the results according to the task
        """
        pass