"""
@authors: Filip Maciejewski, Oskar Słowik, Tomek Rybotycki
@contact: filip.b.maciejewski@gmail.com

REFERENCES:
[0] Filip B. Maciejewski, Zoltán Zimborás, Michał Oszmaniec,
"Mitigation of readout noise in near-term quantum devices
by classical post-processing based on detector tomography",
Quantum 4, 257 (2020)

[0.5] Filip B. Maciejewski, Flavio Baccari, Zoltán Zimborás, Michał Oszmaniec,
"Modeling and mitigation of cross-talk effects in readout noise
with applications to the Quantum Approximate Optimization Algorithm",
Quantum 5, 464 (2021).

"""
import abc
from typing import List


class MarginalsAnalyzerInterface(abc.ABC):
    """
    This is interface for classes that will analyze marginal probability distributions.
    It requires those child classes to have basic functionalities that should be included.
    for analyzing marginals_dictionary.
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
        # Updating dictionary with results
        raise NotImplementedError

    @abc.abstractmethod
    def marginals_dictionary_update(self, marginals_dictionary_new: dict) -> None:
        # Updating dictionary with new marginals_dictionary
        raise NotImplementedError

    @abc.abstractmethod
    def compute_marginals(self,
                          experiment_key: str,
                          subsets: List[List[int]]) -> dict:
        """Computes marginals_dictionary for input subsets_list of qubits"""
        raise NotImplementedError
