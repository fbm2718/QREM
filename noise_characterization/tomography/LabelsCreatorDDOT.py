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
import copy
from typing import Optional, List, Dict, Union
import numpy as np
from QREM.noise_characterization.tomography.LabelsBaseDDOT import LabelsBaseDDOT
from functions import ancillary_functions as anf
from tqdm import tqdm
import time
from scipy.special import binom as binomial_coefficient


class LabelsCreatorDDOT(LabelsBaseDDOT):
    """
    Class for creating symbolic description of Diagonal Detector Overlapping Tomography (DDOT)
    circuits.

    Description of a single circuit is simply a bitstring denoting classical input state.


    """

    def __init__(self,
                 number_of_qubits: int,
                 maximal_circuits_amount: Optional[int] = 1500,
                 subsets_list: Optional[List[List[int]]] = None,
                 subsets_locality: Optional[int] = None,
                 show_progress_bars: Optional[bool] = True
                 ):

        """

        :param number_of_qubits:
        :param subsets_list: list of subsets of qubits for which marginal tomography
                             should be performed
                             NOTE: for tomography which is to be performed NOT on ALL subsets
                             of given locality, this must be provided

        :param subsets_locality: locality of ALL considered subsets
                                 NOTE: this should be provided only if ALL subsets have the same
                                       locality

        """

        if subsets_list is None and subsets_locality is None:
            raise ValueError('Please provide subsets list or desired locality')
        elif subsets_list is None and subsets_locality is not None:
            subsets_list = anf.get_k_local_subsets(number_of_qubits=number_of_qubits,
                                                   locality=subsets_locality)

        super().__init__(number_of_qubits=number_of_qubits,
                         subsets=subsets_list,
                         maximal_circuits_amount=maximal_circuits_amount,
                         show_progress_bars=show_progress_bars

                         )

        self._subsets_locality = subsets_locality
        if subsets_locality is not None:
            self._elements_in_perfect_collection = int(
                binomial_coefficient(self._number_of_qubits,
                                     self._subsets_locality) *
                2 ** self._subsets_locality)
        else:
            self._elements_in_perfect_collection = int(sum(
                [2 ** len(subset_now) for subset_now in subsets_list]))

        dictionary_symbols_counting = {
            self.get_qubits_key(subset_list): np.zeros(2 ** len(subset_list),
                                                       dtype=int) for
            subset_list in self._subsets}

        self._dictionary_symbols_counting = dictionary_symbols_counting

        circuits_properties_dictionary = {'number_of_circuits': 0,
                                          'absent_elements_amount': 10 ** 6,
                                          'minimal_amount': 10 ** 6,
                                          'maximal_amount': 10 ** 6,
                                          'median': 10 ** 6,
                                          'mean': 10 ** 6,
                                          'SD': 10 ** 6}

        self._circuits_properties_dictionary = circuits_properties_dictionary

    @property
    def dictionary_symbols_counting(self) -> Dict[str, np.ndarray]:
        return self._dictionary_symbols_counting

    @property
    def circuits_properties_dictionary(self) -> Dict[str, float]:
        return self._circuits_properties_dictionary

    @property
    def circuits_list(self) -> List[List[int]]:
        return self._circuits_list

    @circuits_list.setter
    def circuits_list(self, circuits_list: List[List[int]]):
        self.reset_everything()
        self._circuits_list = circuits_list

    def add_dictionary_subsets_symbols_counting_template(self):
        dictionary_symbols_counting = {
            self.get_qubits_key(subset_list): np.zeros(2 ** len(subset_list),
                                                       dtype=int) for
            subset_list in self._subsets}

        self._dictionary_symbols_counting = dictionary_symbols_counting

    def update_dictionary_subsets_symbols_counting(self,
                                                   circuits_list: Optional[List[int]] = None,
                                                   count_added_subcircuits: Optional[bool] = False):
        if circuits_list is None:
            circuits_list = self._circuits_list

        subsets_range = range(len(self._subsets))
        if self._show_progress_bars:
            subsets_range = tqdm(subsets_range)

        if count_added_subcircuits:
            added_subcircuits_counter = 0
        else:
            added_subcircuits_counter = None

        for subset_index in subsets_range:
            for circuit in circuits_list:

                qubits_key = self.get_qubits_key(self._subsets[subset_index])
                subset_circuit_identifier = int(
                    ''.join([str(circuit[qubit_index]) for qubit_index in
                             self._subsets[subset_index]]),2)

                if count_added_subcircuits:
                    if self._dictionary_symbols_counting[qubits_key][subset_circuit_identifier] == 0:
                        added_subcircuits_counter += 1

                self._dictionary_symbols_counting[qubits_key][subset_circuit_identifier] += 1

        if count_added_subcircuits:
            return added_subcircuits_counter

    def get_absent_symbols_amount(self):

        t0 = time.time()
        zero_subsets = 0
        for subset in self._subsets:
            subset_counts = self._dictionary_symbols_counting[self.get_qubits_key(subset)]
            zero_subsets += len(subset_counts) - np.count_nonzero(subset_counts)

        self._circuits_properties_dictionary['absent_elements_amount'] = zero_subsets
        anf.cool_print('This took:', time.time() - t0)

        return zero_subsets

    def calculate_properties_of_circuits(self) -> None:

        big_list = []
        for subset in self._subsets:
            big_list += list(self._dictionary_symbols_counting[self.get_qubits_key(subset)])

        minimal_amount, maximal_amount = min(big_list), max(big_list)

        big_list = np.array(big_list)

        mean, SD, median = np.mean(big_list), np.std(big_list), np.median(big_list)

        absent_elements_amount = len(big_list) - np.count_nonzero(big_list)

        self._circuits_properties_dictionary['number_of_circuits'] = len(self._circuits_list)
        self._circuits_properties_dictionary['absent_elements_amount'] = absent_elements_amount
        self._circuits_properties_dictionary['minimal_amount'] = minimal_amount
        self._circuits_properties_dictionary['maximal_amount'] = maximal_amount
        self._circuits_properties_dictionary['median'] = median
        self._circuits_properties_dictionary['mean'] = mean
        self._circuits_properties_dictionary['SD'] = SD

    def reset_everything(self):
        circuits_properties_dictionary = {'number_of_circuits': 0,
                                          'absent_elements_amount': 10 ** 6,
                                          'minimal_amount': 10 ** 6,
                                          'maximal_amount': 10 ** 6,
                                          'median': 10 ** 6,
                                          'mean': 10 ** 6,
                                          'SD': 10 ** 6}

        self._circuits_properties_dictionary = circuits_properties_dictionary
        self._circuits_list = []

        dictionary_symbols_counting = {
            self.get_qubits_key(subset_list): np.zeros(2 ** len(subset_list),
                                                       dtype=int) for
            subset_list in self._subsets}

        self._dictionary_symbols_counting = dictionary_symbols_counting

    def _cost_function_circuits_amount(self):
        return len(self._circuits_list)

    def _cost_function_circuits_SD(self):
        return self._circuits_properties_dictionary['SD']

    def _cost_function_minimal_amount_of_circuits(self):
        return -self._circuits_properties_dictionary['minimal_amount']

    def _cost_function_absent_elements(self):
        return self._circuits_properties_dictionary['absent_elements_amount']

    def _cost_function_maximal_spread(self):
        return self._circuits_properties_dictionary['maximal_amount'] - \
               self._circuits_properties_dictionary['minimal_amount']

    def _compute_perfect_collection_bruteforce(self,
                                               circuits_in_batch: int,
                                               print_updates: bool
                                               ):

        runs_number = 1
        absent_elements_amount = self._elements_in_perfect_collection

        while absent_elements_amount > 0 and runs_number < self._maximal_circuits_amount:
            if runs_number % 20 == 0 and print_updates:
                anf.cool_print('Run number:', runs_number)
                anf.cool_print('Number of circuits:', len(self._circuits_list))
                anf.cool_print('Absent elements amount:', absent_elements_amount)

            circuits_now = self.get_random_circuits(circuits_in_batch)
            self.add_circuits(circuits_now)

            added_elements = self.update_dictionary_subsets_symbols_counting(
                circuits_list=circuits_now,
                count_added_subcircuits=True)
            absent_elements_amount -= added_elements

            runs_number += 1

    def _get_proper_cost_function(self,
                                  optimized_quantity: str):

        if optimized_quantity in ['circuits_amount', 'circuits_number', 'amount', 'circuits']:
            cost_function = self._cost_function_circuits_amount
        elif optimized_quantity in ['std', 'SD', 'standard_deviation']:
            cost_function = self._cost_function_circuits_SD
        elif optimized_quantity in ['minimal_amount']:
            cost_function = self._cost_function_minimal_amount_of_circuits
        elif optimized_quantity in ['spread', 'maximal_spread']:
            cost_function = self._cost_function_maximal_spread
        elif optimized_quantity in ['absent', 'absent_elements']:
            cost_function = self._cost_function_absent_elements
        else:
            raise ValueError('Wrong optimized quantity string: ' + optimized_quantity + '.')

        return cost_function

    def _add_cost_functions(self,
                            dictionary_cost_functions: Dict[str, float]):

        def cost_functions_added():
            returned_quantity = 0
            for function_label, function_weight in dictionary_cost_functions.items():
                returned_quantity += function_weight * self._get_proper_cost_function(function_label)()
            return returned_quantity

        return cost_functions_added

    def _compute_perfect_collection_bruteforce_randomized(self,
                                                          number_of_iterations: int,
                                                          circuits_in_batch: int,
                                                          print_updates: bool,
                                                          optimized_quantity: Union[
                                                              str, Dict[str, float]],
                                                          additional_circuits: Optional[int] = 0
                                                          ):
        """
        This function implements self._compute_perfect_collection_bruteforce
        for number_of_iterations times, then adds additional_circuits number of random circuits,
        computes cost function and chooses the family that minimizes cost function.

        :param number_of_iterations: how many times random perfect family should be generated
        :param circuits_in_batch: see self._compute_perfect_collection_bruteforce
        :param print_updates: whether to print updates during optimization
        :param optimized_quantity: specify what cost function is
        Possible string values:
        1. 'minimal_amount' - maximizes number of least-frequent subset-circuits
        2. 'spread' - minimizes difference between maximal and minimal number of subset-circuits

        3. 'circuits_amount' - minimizes number of circuits
                        (NOTE: it does not make sense to choose this option with additional_circuits>0)
        4. 'SD' - minimizes standard deviation of occurrences of subset-circuits

        It is possible to use combined cost functions.
        Dictionary must be provided where KEY is label for optimized quantity and VALUE is its weight.

        For example:
        optimized_quantity = {'minimal_amount': 1.0,
                              'spread':0.5}

        will give cost function which returns 1.0 * (-number of least frequent circuits)
                                          + 0.5 * (difference between most and least fequenet circuits)


        :param additional_circuits: number of circuits which are to be added to the PERFECT collection
                                    obtained in optimization loop. Those are "additional" circuits in
                                    a sense that they are not needed for collection to be perfect,
                                    but instead are used to obtain better values of cost function
                                    or just add more experiments reduce statistical noise.
        :return:
        """

        if isinstance(optimized_quantity, str):
            cost_function = self._get_proper_cost_function(optimized_quantity=optimized_quantity)
        elif isinstance(optimized_quantity, dict):
            cost_function = self._add_cost_functions(dictionary_cost_functions=optimized_quantity)

        runs_range = range(number_of_iterations)
        if self._show_progress_bars:
            runs_range = tqdm(runs_range)

        # circuit_families = []
        # best_family = None
        global_cost, best_family = 10 ** 6, None

        for runs_number in runs_range:
            if runs_number % int(np.ceil(number_of_iterations / 20)) == 0 and print_updates:
                anf.cool_print('Run number:', runs_number, 'red')
                anf.cool_print('Current best value of cost function:', global_cost)

            self.reset_everything()
            self._compute_perfect_collection_bruteforce(circuits_in_batch=circuits_in_batch,
                                                        print_updates=False
                                                        )

            if additional_circuits > 0:
                current_length, maximal_length = len(
                    self._circuits_list), self._maximal_circuits_amount
                if current_length < maximal_length:
                    if additional_circuits > maximal_length - current_length:
                        adding_circuits = maximal_length - current_length
                    else:
                        adding_circuits = copy.deepcopy(additional_circuits)
                    # print(adding_circuits)
                    new_circuits = self.get_random_circuits(adding_circuits)

                    self.update_dictionary_subsets_symbols_counting(new_circuits)
                    self.add_circuits(new_circuits)

            self.calculate_properties_of_circuits()

            cost_now = cost_function()

            if cost_now < global_cost:
                best_family = copy.deepcopy(self._circuits_list)
                global_cost = cost_now

        anf.cool_print('best family length', len(best_family), 'red')
        self.reset_everything()
        self._circuits_list = best_family
        self.update_dictionary_subsets_symbols_counting()
        self.calculate_properties_of_circuits()

    def compute_perfect_collection(self,
                                   method_name='bruteforce_randomized',
                                   method_kwargs=None):

        """
        Find perfect collection of overlapping circuits.
        "Perfect" means that for each subset of qubits self._subsets_list[i],
         each symbol out of self._number_of_symbols^self._subsets_list[i],
         appears in the collection at least once.

        :param method_name:
        possible values:

        1. 'bruteforce' - see self._compute_perfect_collection_bruteforce
        2. 'bruteforce_randomized' - see self._compute_perfect_collection_bruteforce_randomized

        :param method_kwargs: kwargs for chosen method, see corresponding methods' descriptions
        :return:
        """

        if method_name == 'bruteforce':
            if method_kwargs is None:
                method_kwargs = {'circuits_in_batch': 1,
                                 'print_updates': True}

            self._compute_perfect_collection_bruteforce(**method_kwargs)

        elif method_name == 'bruteforce_randomized':
            if method_kwargs is None:
                method_kwargs = {'number_of_iterations': 100,
                                 'circuits_in_batch': 1,
                                 'print_updates': True,
                                 'optimized_quantity': 'minimal_amount',
                                 'additional_circuits': 0}

            self._compute_perfect_collection_bruteforce_randomized(**method_kwargs)

        absent_elements = self._circuits_properties_dictionary['absent_elements_amount']
        if absent_elements != 0:
            anf.cool_print('________WARNING________:',
                           'The collection is not perfect. '
                           'It is missing %s' % absent_elements
                           + ' elements!\nTry increasing limit on the circuits amount.',
                           'red')
