"""
Created on 28.04.2021

@author: Filip Maciejewski
@contact: filip.b.maciejewski@gmail.com
"""

"""
    This class should be used as a base class for all standard BS permanent calculators. By standard I mean that the
    matrix and in(out)put states are stored in a variables. It takes care of a lot of boilerplate code.
"""

from typing import Optional
from DDOT_module.parent_classes.marginals_analyzer_interface import MarginalsAnalyzerInterface
import numpy as np
from DDOT_module.functions import functions_data_analysis as fda
from QREM import ancillary_functions as anf
from typing import List


class MarginalsAnalyzerBase(MarginalsAnalyzerInterface):
    def __init__(self,
                 results_dictionary: dict,
                 reverse_counts: bool,
                 marginals_dictionary: Optional[dict] = None,
                 ) -> None:
        if marginals_dictionary is None:
            marginals_dictionary = {key: fda.key_dependent_dict_for_marginals()
                                    for key in results_dictionary.keys()}

        self._results_dictionary = results_dictionary
        self._marginals_dictionary = marginals_dictionary
        self._reverse_counts = reverse_counts

    @property
    def results_dictionary(self) -> dict:
        return self._results_dictionary

    @results_dictionary.setter
    def results_dictionary(self, results_dictionary: dict) -> None:
        self._results_dictionary = results_dictionary

    @property
    def marginals_dictionary(self) -> dict:
        return self._marginals_dictionary

    @marginals_dictionary.setter
    def marginals_dictionary(self, marginals_dictionary: dict) -> None:
        self._marginals_dictionary = marginals_dictionary

    def results_dictionary_update(self,
                                  results_dictionary_new: dict) -> None:
        self._results_dictionary = {**self._results_dictionary,
                                    **results_dictionary_new}

    def marginals_dictionary_update(self,
                                    marginals_dictionary_new: dict) -> None:
        self._marginals_dictionary = {**self._marginals_dictionary,
                                      **marginals_dictionary_new}

    def normalize_marginals(self):
        for key_experiment in self._marginals_dictionary.keys():
            for key_marginal in self._marginals_dictionary[key_experiment].keys():
                self._marginals_dictionary[key_experiment][key_marginal] *= 1 / np.sum(
                    self._marginals_dictionary[key_experiment][key_marginal])

    def compute_marginals(self,
                          experiment_key: str,
                          subsets_list: List[List[int]]) -> None:
        """Return dictionary of marginal probability distributions from counts dictionary
        :param experiment_key: key that labels experiment from which marginals should be taken
        :param subsets_list: list of subsets of qubits for which marginals should be calculated
        """
        experimental_results = self._results_dictionary[experiment_key]

        for subset in subsets_list:
            marginal_vector_now = np.zeros((int(2 ** len(subset)), 1),
                                           dtype=float)

            for count, ticks in experimental_results.items():
                if self._reverse_counts:
                    count = count[::-1]

                marginal_key_now = ''.join([count[b] for b in subset])
                marginal_vector_now[int(marginal_key_now, 2)] += ticks

            qubits_string = 'q' + 'q'.join([str(s) for s in subset])
            self._marginals_dictionary[experiment_key][qubits_string] += marginal_vector_now

        self.normalize_marginals()

    def compute_all_marginals(self,
                              subsets_list: List[List[int]],
                              show_progress_bar: Optional[bool] = False) -> None:
        """
        Implements self.compute_marginals for all experimental keys.

        :param subsets_list: list of subsets of qubits for which marginals should be calculated
        :param show_progress_bar: if True, shows progress bar. Requires "tqdm" package
        """

        if show_progress_bar:
            from tqdm import tqdm
            keys_list = list(self._results_dictionary.keys())
            for key_index in tqdm(range(len(keys_list))):
                self.compute_marginals(keys_list[key_index], subsets_list)

        else:
            for key in self._results_dictionary.keys():
                self.compute_marginals(key, subsets_list)

    def get_marginals(self,
                      experiment_key: str,
                      subsets_list: List[List[int]]) -> dict:
        """Like self.compute_marginals but first checks if the marginals are already computed and
           returns them.

        :param experiment_key: key that labels experiment from which marginals should be taken
        :param subsets_list: list of subsets of qubits for which marginals should be calculated

        :return: marginals_dictionary (dictionary):
                dictionary in which KEY is label for experiment, and VALUE is dictionary with KEYS
                being qubit subset identifiers (in a format "q1q5q23" etc.), and VALUEs being marginal
                probability vectors
        """

        keys_list = ['q' + 'q'.join([str(s) for s in subset])
                     for subset in subsets_list]

        for i in range(len(subsets_list)):
            subset, key_now = subsets_list[i], keys_list[i]

            if experiment_key not in self._marginals_dictionary.keys():
                self.compute_marginals(experiment_key, [subset])
            elif key_now not in self._marginals_dictionary[experiment_key].keys():
                self.compute_marginals(experiment_key, [subset])

        return {key_now: self._marginals_dictionary[experiment_key][key_now]
                for key_now in keys_list}

    @staticmethod
    def get_marginal_from_probability_distribution(
            global_probability_distribution: np.ndarray,
            bits_of_interest: List[int],
            register_names: Optional[List[str]] = None) -> np.ndarray:

        """Return marginal distribution from vector of global distribution
        :param global_probability_distribution: distribution on all bits
        :param bits_of_interest: bits we are interested in (so we average over other bits)
        :param register_names: bitstrings register names, default is
                               '00...00', '000...01', '000...10', ..., etc.

        :return: marginal_distribution : marginal probability distribution
        """

        if len(bits_of_interest) == 0:
            print('0 length bits list')
            return global_probability_distribution

        d = len(global_probability_distribution)
        n = int(np.log2(d))
        qubits = list(range(n))
        bits_to_average_over = list(set(qubits) - set(bits_of_interest))

        n_post = n - len(bits_to_average_over)
        d_post = 2 ** n_post

        if register_names is None:
            names = anf.register_names_qubits(range(n), n)
        else:
            names = register_names

        marginal_distribution = np.zeros(d_post)
        rest = list(range(n))

        for x in bits_to_average_over:
            rest.remove(x)

        for j in range(d):
            name = names[j]
            s = ''
            for q in rest:
                s += name[q]

            ind = int(s, 2)
            marginal_distribution[ind] += global_probability_distribution[j]

        marginal_distribution = np.array(marginal_distribution).reshape(2 ** len(bits_of_interest), 1)

        return marginal_distribution
