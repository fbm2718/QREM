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
    @property
    @abc.abstractmethod
    def results_dictionary(self) -> dict:
        '''Results dictionary for which KEY is the bitstring denoting LABEL OF EXPERIMENT,
         while VALUE is the counts dictionary with results of the experiments'''

        raise NotImplementedError

    @results_dictionary.setter
    @abc.abstractmethod
    def results_dictionary(self, results_dictionary: dict) -> None:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def marginals_dictionary(self) -> dict:
        '''Marginals dictionary for which each KEY is label of the subset, and each VALUE is
           dictionary of those marginals'''
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
        raise NotImplementedError
    @abc.abstractmethod
    def marginals_dictionary_update(self, marginals_dictionary_new: dict) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def compute_marginals(self,
                          experiment_key: str,
                          subsets: list) -> dict:
        """Computes marginals for input subsets"""
        raise NotImplementedError






