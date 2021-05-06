"""
Created on 28.04.2021

@author: fbm
@contact: filip.b.maciejewski@gmail.com
"""
import abc

"""
    This file holds an interface for Marginals Analyzers. BS. 
"""


class MarginalsAnalyzerInterface(abc.ABC):
    """
    This is interface for classes that will analyze marginal probability distributions.
    It requires those child classes to have basic functionalities that should be included.
    for analyzing marginals.
    """

    @property
    @abc.abstractmethod
    def results_dictionary(self) -> dict:
        # dictionary of experimental results

        raise NotImplementedError

    @results_dictionary.setter
    @abc.abstractmethod
    def results_dictionary(self, results_dictionary: dict) -> None:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def marginals_dictionary(self) -> dict:
        # Dictionary storing marginal probability distributions
        raise NotImplementedError

    @results_dictionary.setter
    @abc.abstractmethod
    def results_dictionary(self, results_dictionary: dict) -> None:
        raise NotImplementedError

    @marginals_dictionary.setter
    @abc.abstractmethod
    def marginals_dictionary(self, marginals_dictionary: dict) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def results_dictionary_update(self, results_dictionary_new: dict) -> None:
        #Updating dictionary with results
        raise NotImplementedError

    @abc.abstractmethod
    def marginals_dictionary_update(self, marginals_dictionary_new: dict) -> None:
        #Updating dictionary with new marginals
        raise NotImplementedError

    @abc.abstractmethod
    def compute_marginals(self,
                          experiment_key: str,
                          subsets: list) -> dict:
        """Computes marginals for input subsets"""
        raise NotImplementedError
