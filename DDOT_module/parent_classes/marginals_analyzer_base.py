"""
Created on 28.04.2021

@author: Filip Maciejewski
@contact: filip.b.maciejewski@gmail.com
"""
from typing import Optional, Dict
from DDOT_module.parent_classes.marginals_analyzer_interface import MarginalsAnalyzerInterface
import numpy as np
from DDOT_module.functions import functions_data_analysis as fda
from QREM import ancillary_functions as anf
from typing import List
import re


class MarginalsAnalyzerBase(MarginalsAnalyzerInterface):
    '''
    This is base class for all the classes that will operate on marginal probability distributions.
    Methods of this class allow to calculate marginal distributions from experimental results.
    Please note how we format results of experiments and marginal distributions
    in the description of __init__.
    '''

    def __init__(self,
                 results_dictionary: Dict[str, Dict[str, int]],
                 bitstrings_right_to_left: bool,
                 marginals_dictionary: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
                 ) -> None:

        """
        Description:
            This is default constructor for MarginalsAnalyzerBase objects.
            It requires to provide results of experiments and information whether bitstrings should be
            read from right to left (when interpreting qubit labels)

        :param results_dictionary: Nested dictionary with following structure:

        results_dictionary[label_of_experiment][bitstring_outcome] = number_of_occurrences

        where:
        -label_of_experiment is arbitrary label for particular experiment,
        -bitstring_outcome is label for measurement outcome,
        -number_of_occurrences is number of times that bitstring_outcome was recorded

        Hence top-level key labels particular experiment
        (one can think about quantum circuit implementation)
        and its value is another dictionary with results of given experiment in the form
        of dictionary of measurement outcomes


        :param bitstrings_right_to_left: specify whether bitstrings
                                    should be read from right to left (when interpreting qubit labels)
        :param marginals_dictionary: Nested dictionary with the following structure:

        marginals_dictionary[label_of_experiment][label_of_subset] = marginal_probability_vector

        where:
        -label_of_experiment is the same as in results_dictionary and it labels results from which
        marginal distributions were calculated
        -label_of_subset is a label for qubits subset for which marginals were calculated.
        We use convention that such label if of the form "q5q8q12..." etc., hence it is bitsring of
        qubits labels starting from "q".
        -marginal_probability_vector marginal distribution stored as vector

        NOTE: when user does not provide marginals_dictionary we create it during class initialization.
        To this aim, we create "key_dependent_dicts"


        """

        if marginals_dictionary is None:
            # If user does not provide dictionary with marginals, we create template.
            # Instead of standard dictionaries, we use ones that are "key dependent" (see description
            # of that function), which is useful for some calculations.
            marginals_dictionary = {key: fda.KeyDependentDictForMarginals()
                                    for key in results_dictionary.keys()}

        # set initial values of class properties
        self._results_dictionary = results_dictionary
        self._marginals_dictionary = marginals_dictionary
        self._bitstrings_right_to_left = bitstrings_right_to_left

    @property
    def results_dictionary(self) -> Dict[str, Dict[str, int]]:
        return self._results_dictionary

    @results_dictionary.setter
    def results_dictionary(self, results_dictionary: Dict[str, Dict[str, int]]) -> None:
        self._results_dictionary = results_dictionary

    @property
    def marginals_dictionary(self) -> Dict[str, Dict[str, np.ndarray]]:
        return self._marginals_dictionary

    @marginals_dictionary.setter
    def marginals_dictionary(self, marginals_dictionary: Dict[str, Dict[str, np.ndarray]]) -> None:
        self._marginals_dictionary = marginals_dictionary

    @staticmethod
    def get_qubits_key(list_of_qubits: List[int]) -> str:
        """ from subset of qubit indices get the string that labels this subset
            using convention 'q5q6q12...' etc.
        :param list_of_qubits: labels of qubits

        :return: string label for qubits

         NOTE: this function is "dual" to self.get_qubit_indices_from_string.
        """

        return 'q' + 'q'.join([str(s) for s in list_of_qubits])

    @staticmethod
    def get_qubit_indices_from_string(qubits_string: str,
                                      with_q: Optional[bool] = False):

        """Return list of qubit indices from the string of the form "q0q1q22q31"
        :param qubits_string: string which has the form of "q" followed by qubit index
        :param with_q: specify whether returned indices should be in form of string with letter

        :return: list of qubit indices:

        depending on value of parameter "with_q" the mapping will be one of the following:
        if with_q:
            'q1q5q13' -> ['q1','q5','q13']
        else:
            'q1q5q13' -> [1,5,13]

        NOTE: this function is "dual" to self.get_qubits_key.
        """

        numbers = re.findall(r'\d+', qubits_string)

        if with_q:
            qubits = ['q' + s for s in numbers]
        else:
            qubits = [int(s) for s in numbers]

        return qubits

    def results_dictionary_update(self,
                                  results_dictionary_new: Dict[str, Dict[str, int]]) -> None:
        # This method updates results dictionary from class property with new dictionary.
        # Note that if there is KEY collision, then the value from new dictionary overwrites old one.

        self._results_dictionary = {**self._results_dictionary,
                                    **results_dictionary_new}

    def marginals_dictionary_update(self,
                                    marginals_dictionary_new: Dict[
                                        str, Dict[str, np.ndarray]]) -> None:
        # See description of self.results_dictionary_update

        self._marginals_dictionary = {**self._marginals_dictionary,
                                      **marginals_dictionary_new}

    def normalize_marginals(self):
        # Loop through all experiments and marginals and normalize them.
        for key_experiment in self._marginals_dictionary.keys():
            for key_marginal in self._marginals_dictionary[key_experiment].keys():
                self._marginals_dictionary[key_experiment][key_marginal] *= 1 / np.sum(
                    self._marginals_dictionary[key_experiment][key_marginal])

    def compute_marginals(self,
                          experiment_key: str,
                          subsets_list: List[List[int]]) -> None:
        """Return dictionary of marginal probability distributions from counts dictionary
        :param experiment_key: key that labels experiment for which marginals should be taken
        :param subsets_list: list of subsets of qubits for which marginals should be calculated
        """
        experimental_results = self._results_dictionary[experiment_key]

        for subset in subsets_list:
            # initialize marginal distribution
            marginal_vector_now = np.zeros((int(2 ** len(subset)), 1),
                                           dtype=float)

            for outcome_bitstring, number_of_occurrences in experimental_results.items():
                if self._bitstrings_right_to_left:
                    # here we change the order of bitstring if it was specified
                    outcome_bitstring = outcome_bitstring[::-1]

                # get bitstring denoting state of qubits in the subset
                marginal_key_now = ''.join([outcome_bitstring[b] for b in subset])

                # add counts to the marginal distribution
                marginal_vector_now[int(marginal_key_now, 2)] += number_of_occurrences

            qubits_string = self.get_qubits_key(subset)

            # Here if there is no "qubits_string" KEY we use the fact that defaultly we use
            # "key_dependent_dictionary". See description of __init__.
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

        keys_list = list(self._results_dictionary.keys())
        keys_list_range = range(len(keys_list))

        # if progress bar should be shown, we use tqdm package
        if show_progress_bar:
            from tqdm import tqdm
            keys_list_range = tqdm(keys_list_range)

        for key_index in keys_list_range:
            self.compute_marginals(keys_list[key_index], subsets_list)

    def get_marginals(self,
                      experiment_key: str,
                      subsets_list: List[List[int]]) -> Dict[str, Dict[str, np.ndarray]]:
        """Like self.compute_marginals but first checks if the marginals are already computed
            and it returns them.

        :param experiment_key: key that labels experiment from which marginals should be taken
        :param subsets_list: list of subsets of qubits for which marginals should be calculated

        :return: marginals_dictionary:
                dictionary in which KEY is label for experiment, and VALUE is dictionary with KEYS
                being qubit subset identifiers (in a format "q1q5q23" etc.), and VALUES being marginal
                probability vectors (see __init__ description)
        """

        keys_list = [self.get_qubits_key(subset)
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
                                Assuming that qubits are labeled
                                from 0 to log2(len(global_probability_distribution))
        :param register_names: bitstrings register bistring_names, default is
                               '00...00', '000...01', '000...10', ..., etc.

        :return: marginal_distribution : marginal probability distribution

        NOTE: we identify bits with qubits in the variables bistring_names

        #TODO FBM: do some speed tests on some details of those solutions
        """

        if len(bits_of_interest) == 0:
            print('0 length bits list')
            return global_probability_distribution

        global_dimension = len(global_probability_distribution)
        global_number_of_qubits = int(np.log2(global_dimension))
        all_qubits = list(range(global_number_of_qubits))
        bits_to_average_over = list(set(all_qubits).difference(set(bits_of_interest)))

        number_of_bits_in_marginal = global_number_of_qubits - len(bits_to_average_over)
        dimension_of_marginal = 2 ** number_of_bits_in_marginal

        if register_names is None:
            bistring_names = anf.register_names_qubits(range(global_number_of_qubits),
                                                       global_number_of_qubits)
        else:
            bistring_names = register_names

        marginal_distribution = np.zeros((dimension_of_marginal, 1), dtype=float)
        for j in range(global_dimension):
            # this is slightly faster than going through "for bitstring_global in bitstring_names
            # and then converting bitstring_global to integer
            # and also faster than creating the global bitstring in situ
            bitstring_global = bistring_names[j]

            bitstring_local = ''.join(
                [bitstring_global[qubit_index] for qubit_index in bits_of_interest])

            marginal_distribution[int(bitstring_local, 2)] += global_probability_distribution[j]

        return marginal_distribution
