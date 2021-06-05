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
import numpy as np
from typing import Optional, Dict, List, Union
from collections import defaultdict
from base_classes.marginals_analyzer_base import MarginalsAnalyzerBase
from functions import ancillary_functions as anf


class DDTMarginalsAnalyzer(MarginalsAnalyzerBase):
    """
        Class that handles results of Diagonal Detector Tomography.
        Main functionalities allow to calculate noise matrices on subsets_list of qubits.
        This includes averaged noise matrices, i_index.e., averaged over states off all other qubits,
        as well as state-dependent, i_index.e., conditioned on the particular
        input classical state of some other qubits.

        In this class, and all its children, we use the following convention
        for storing marginal noise matrices:

        :param noise_matrices_dictionary: nested dictionary with following structure:

        noise_matrices_dictionary[qubits_subset_string]['averaged']
        = average noise matrix on qubits subset
        and
        noise_matrices_dictionary[qubits_subset_string][other_qubits_subset_string][input_state_bitstring]
        = noise matrix on qubits subset depending on input state of other qubits.

        where:
        - qubits_subset_string - is string labeling qubits subset (e.g., 'q1q2q15...')
        - other_qubits_subset_string - string labeling other subset
        - input_state_bitstring - bitstring labeling
                                                   input state of qubits in other_qubits_subset_string

    """

    def __init__(self,
                 results_dictionary_ddot: Dict[str, Dict[str, int]],
                 bitstrings_right_to_left: bool,
                 marginals_dictionary: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
                 noise_matrices_dictionary: Optional[
                     Dict[str, Union[np.ndarray, Dict[str, Dict[str, np.ndarray]]]]] = None
                 ) -> None:

        """
        :param results_dictionary_ddot: see description of MarginalsAnalyzerBase.
        Here we use classical input states (bitstrings) of qubits as LABELS for experiments.
        :param bitstrings_right_to_left: specify whether bitstrings
                                 should be read from right to left (when interpreting qubit labels)
        :param marginals_dictionary: see description of MarginalsAnalyzerBase.
        :param noise_matrices_dictionary: nested dictionary with following structure:
        """

        super().__init__(results_dictionary_ddot,
                         bitstrings_right_to_left,
                         marginals_dictionary
                         )

        if noise_matrices_dictionary is None:
            noise_matrices_dictionary = {}
            # TODO FBM: Make sure whether this helps in anything
            # TODO FBM: (because we anyway perform checks in the functions later)
            if marginals_dictionary is not None:
                for experiment_key, dictionary_of_marginals in marginals_dictionary.items():
                    for marginal_key in dictionary_of_marginals.keys():
                        if marginal_key not in noise_matrices_dictionary.keys():
                            noise_matrices_dictionary[marginal_key] = {}

        self._noise_matrices_dictionary = noise_matrices_dictionary

    @property
    def noise_matrices_dictionary(self) -> Dict[str, Union[
        np.ndarray, Dict[str, Dict[str, np.ndarray]]]]:
        return self._noise_matrices_dictionary

    @noise_matrices_dictionary.setter
    def noise_matrices_dictionary(self,
                                  noise_matrices_dictionary: Dict[str, Union[
                                      np.ndarray, Dict[str, Dict[str, np.ndarray]]]] = None) -> None:

        self._noise_matrices_dictionary = noise_matrices_dictionary

    @staticmethod
    def get_noise_matrix_from_counts_dict(
            results_dictionary: Union[Dict[str, np.ndarray], defaultdict]) -> np.ndarray:
        """Return noise matrix from counts dictionary.
        Assuming that the results are given only for qubits of interest.
        :param results_dictionary: dictionary with experiments of the form:

        results_dictionary[input_state_bitstring] = probability_distribution

        where:
        - input_state_bitstring is bitstring denoting classical input state
        - probability_distribution - estimated vector of probabilities for that input state

        :return: noise_matrix: the array representing noise on qubits
        on which the experiments were performed
        """

        number_of_qubits = len(list(results_dictionary.keys())[0])
        noise_matrix = np.zeros((2 ** number_of_qubits, 2 ** number_of_qubits))
        for input_state, probability_vector in results_dictionary.items():
            noise_matrix[:, int(input_state, 2)] = probability_vector[:, 0]
        return noise_matrix

    @staticmethod
    def average_noise_matrices_over_some_qubits(matrices_cluster: Dict[str, np.ndarray],
                                                all_neighbors: List[int],
                                                qubits_to_be_left: List[int]) -> Dict[str, np.ndarray]:
        """
          Given dictionary of noise matrices, average them over some qubits.

         :param matrices_cluster: dictionary for which KEY is classical INPUT state of neighbors,
                                  and VALUE is potentially_stochastic_matrix noise matrix
         :param all_neighbors: list of neighbors of given cluster
         :param qubits_to_be_left: qubits which we are interested in and we do not average over them

         :return: dictionary of noise matrices
                    depending on the state of neighbors MINUS qubits_to_be_averaged_over

         """

        if all_neighbors is None or len(all_neighbors)==0:
            return {'averaged': matrices_cluster['averaged']}

        reversed_enumerated = anf.get_reversed_enumerated_from_indices(all_neighbors)
        averaging_normalization = int(2 ** (len(all_neighbors) - len(qubits_to_be_left)))

        states_after_averaging = anf.register_names_qubits(range(len(qubits_to_be_left)),
                                                           len(qubits_to_be_left), False)
        averaged_dimension = list(matrices_cluster.values())[0].shape[0]

        averaged_matrices_cluster = {
            state: np.zeros((averaged_dimension, averaged_dimension), dtype=float) for state in
            states_after_averaging}

        qubits_to_be_averaged_over = list(set(all_neighbors).difference(set(qubits_to_be_left)))
        qubits_to_be_averaged_over_mapped = [reversed_enumerated[q_index] for q_index in
                                             qubits_to_be_averaged_over]

        for neighbors_state, conditional_noise_matrix in matrices_cluster.items():
            list_string_neighbors = list(copy.deepcopy(neighbors_state))

            list_string_neighbors_to_be_left = list(np.delete(list_string_neighbors,
                                                              qubits_to_be_averaged_over_mapped))

            string_neighbors = ''.join(list_string_neighbors_to_be_left)

            averaged_matrices_cluster[
                string_neighbors] += conditional_noise_matrix / averaging_normalization

        return averaged_matrices_cluster

    def _compute_noise_matrix_averaged(self,
                                       subset: List[int]) -> np.ndarray:
        """Noise matrix for subset of qubits, averaged over all other qubits

            :param subset: subset of qubits we are interested in

           By default takes data from self._marginals_dictionary. If data is not present, then it
           calculates marginals_dictionary for given subset
           and updates the class's property self.marginals_dictionary
        """

        # TODO FBM: Perhaps add possibility of using existing marginals_dictionary for bigger subset that includes
        # target subset

        subset_key = 'q' + 'q'.join([str(s) for s in subset])


        marginal_dict_now = self.get_averaged_marginal_for_subset(subset)

        noise_matrix_averaged = self.get_noise_matrix_from_counts_dict(marginal_dict_now)

        if not anf.is_stochastic(noise_matrix_averaged):
            raise ValueError('Noise matrix not stochastic for subset:', subset)

        if subset_key in self._noise_matrices_dictionary.keys():
            self._noise_matrices_dictionary[subset_key]['averaged'] = noise_matrix_averaged
        else:
            self._noise_matrices_dictionary[subset_key] = {'averaged': noise_matrix_averaged}

        return noise_matrix_averaged

    def _get_noise_matrix_averaged(self,
                                   subset: List[int]) -> np.ndarray:
        """
            Like self._compute_noise_matrix_averaged but if matrix is already in class' property,
            does not calculate it again.

            :param subset: subset of qubits we are interested in
        """
        subset_key = 'q' + 'q'.join([str(s) for s in subset])
        try:
            return self._noise_matrices_dictionary[subset_key]['averaged']
        except(KeyError):
            return self._compute_noise_matrix_averaged(subset)

    def _compute_noise_matrix_dependent(self,
                                        qubits_of_interest: List[int],
                                        neighbors_of_interest: Union[List[int], None]) \
            -> Dict[str, np.ndarray]:
        """Return lower-dimensional effective noise matrices acting on qubits_of_interest"
                    conditioned on input states of neighbors_of_interest
            :param qubits_of_interest: labels of qubits in marginal  we are interested in
            :param neighbors_of_interest: labels of qubits that affect noise matrix on qubits_of_interest

            :return conditional_noise_matrices_dictionary: dictionary with structure

            conditional_noise_matrices_dictionary['averaged'] =
            noise matrix on qubits_of_interest averaged over input states of other qubits

            and

            conditional_noise_matrices_dictionary[input_state_neighbors_bitstring] =
            noise matrix on qubits_of_interest conditioned on input state of neighbors being
            input_state_neighbors_bitstring

        """

        # If there are no all_neighbors,
        # then this corresponds to averaging over all qubits except qubits_of_interest
        if len(neighbors_of_interest) == 0 or neighbors_of_interest is None:
            cluster_string = self.get_qubits_key(qubits_of_interest)
            if 'averaged' in self._noise_matrices_dictionary[cluster_string].keys():
                return {'averaged': self._noise_matrices_dictionary[cluster_string]['averaged']}
            else:
                noise_matrix = self._get_noise_matrix_averaged(qubits_of_interest)
                return {'averaged': noise_matrix}

        # check if there is no collision between qubits_of_interest and neighbors_of_interest
        # (if there is, then the method_name won't be consistent)
        if len(anf.lists_intersection(qubits_of_interest, neighbors_of_interest)) != 0:
            print(qubits_of_interest, neighbors_of_interest)
            raise ValueError('Qubits of interest and neighbors overlap')

        # first, get averaged noise matrix on qubits of interest and all_neighbors of interest
        # TODO FBM: make sure that qubit indices are correct (I think they are)
        all_qubits = sorted(qubits_of_interest + neighbors_of_interest)
        all_qubits_enumerated = anf.get_reversed_enumerated_from_indices(all_qubits)

        # we will get noise matrix on all of the qubits first, and then we will process it to get
        # conditional marginal noise matrices on qubits_of_interest
        big_lambda = self._get_noise_matrix_averaged(all_qubits)

        total_number_of_qubits = int(np.log2(big_lambda.shape[0]))
        total_dimension = int(2 ** total_number_of_qubits)
        number_of_qubits_of_interest = len(qubits_of_interest)
        number_of_neighbors = len(neighbors_of_interest)

        # Normalization when averaging over states of non-neighbours (each with the same probability)
        normalization = 2 ** (
                total_number_of_qubits - number_of_neighbors - number_of_qubits_of_interest)

        # classical register on all qubits
        classical_register_all_qubits = ["{0:b}".format(i).zfill(total_number_of_qubits) for i in
                                         range(total_dimension)]

        # classical register on neighbours
        classical_register_neighbours = ["{0:b}".format(i).zfill(number_of_neighbors) for i in
                                         range(2 ** number_of_neighbors)]

        # create dictionary of the marginal states of qubits_of_interest and neighbors_of_interest
        # for the whole register (this function is storing data which could also be calculated in situ
        # in the loops later, but this is faster)
        indices_dictionary_small = {}
        for neighbors_state_bitstring in classical_register_all_qubits:
            small_string = ''.join([list(neighbors_state_bitstring)[all_qubits_enumerated[b]] for b in
                                    qubits_of_interest])
            neighbours_string = ''.join(
                [list(neighbors_state_bitstring)[all_qubits_enumerated[b]] for b in
                 neighbors_of_interest])
            # first place in list is label for state of qubits_of_interest
            # and second for neighbors_of_interest
            indices_dictionary_small[neighbors_state_bitstring] = [small_string, neighbours_string]

        # initiate dictionary for which KEY is input state of all_neighbors
        # and VALUE will the the corresponding noise matrix on qubits_of_interest
        conditional_noise_matrices = {
            s: np.zeros((2 ** number_of_qubits_of_interest, 2 ** number_of_qubits_of_interest)) for s
            in
            classical_register_neighbours}

        # go through all classical states
        for measured_state_integer in range(total_dimension):
            for input_state_integer in range(total_dimension):
                lambda_element = big_lambda[measured_state_integer, input_state_integer]

                # input state of all qubits in binary format
                input_state_bitstring = classical_register_all_qubits[input_state_integer]
                # measured state of all qubits in binary format
                measured_state_bitstring = classical_register_all_qubits[measured_state_integer]

                # input state of qubits_of_interest in binary format
                input_state_small = indices_dictionary_small[input_state_bitstring][0]
                # measured state of qubits_of_interest in binary format
                measured_state_small = indices_dictionary_small[measured_state_bitstring][0]

                # input state of neighbors_of_interest in binary format
                input_state_neighbours = indices_dictionary_small[input_state_bitstring][1]

                # element of small lambda labeled by (measured state | input state),
                # and the lambda itself is labeled by input state of all_neighbors
                conditional_noise_matrices[input_state_neighbours][
                    int(measured_state_small, 2), int(input_state_small, 2)] += lambda_element

        # normalize matrices
        for neighbors_state_bitstring in classical_register_neighbours:
            conditional_noise_matrices[neighbors_state_bitstring] /= normalization

        # conditional_noise_matrices['all_neighbors'] = neighbors_of_interest

        cluster_string = 'q' + 'q'.join(str(s) for s in qubits_of_interest)
        neighbours_string = 'q' + 'q'.join(str(s) for s in neighbors_of_interest)

        if cluster_string not in self._noise_matrices_dictionary.keys():
            # If there is no entry for our cluster in the dictionary, we create it and add
            # averaged noise matrix
            averaged_noise_matrix = np.zeros(
                (2 ** number_of_qubits_of_interest, 2 ** number_of_qubits_of_interest))
            for neighbors_state_bitstring in conditional_noise_matrices.keys():
                averaged_noise_matrix += conditional_noise_matrices[neighbors_state_bitstring]
            averaged_noise_matrix /= 2 ** number_of_qubits_of_interest
            self._noise_matrices_dictionary[cluster_string] = {'averaged': averaged_noise_matrix}

        self._noise_matrices_dictionary[cluster_string][neighbours_string] = conditional_noise_matrices

        return self._noise_matrices_dictionary[cluster_string][neighbours_string]

    def get_noise_matrix_dependent(self,
                                   qubits_of_interest: List[int],
                                   neighbors_of_interest: List[int]) -> dict:
        """Description:
        like self._compute_noise_matrix_dependent
        but checks whether matrices were already calculated to prevent multiple computations of the
        same matrices

        :param qubits_of_interest: labels of qubits in marginal  we are interested in
        :param neighbors_of_interest: labels of qubits that affect noise matrix on qubits_of_interest

        :return conditional_noise_matrices_dictionary:

        """

        cluster_key = self.get_qubits_key(qubits_of_interest)

        if cluster_key not in self._noise_matrices_dictionary.keys():
            self.compute_subset_noise_matrices_averaged([qubits_of_interest])

        if len(neighbors_of_interest) == 0 or neighbors_of_interest is None:
            neighbors_key = 'averaged'

            if neighbors_key in self._noise_matrices_dictionary[cluster_key]:
                if not anf.is_stochastic(self._noise_matrices_dictionary[cluster_key]['averaged']):
                    anf.cool_print('Bug is here')
                    print(cluster_key, neighbors_key)
                    # TODO FBM: SOMETHING IS BROKEN
                    self._noise_matrices_dictionary[cluster_key][
                        'averaged'] = self._compute_noise_matrix_averaged(qubits_of_interest)
                    if not anf.is_stochastic(self._noise_matrices_dictionary[cluster_key]['averaged']):
                        anf.cool_print('And I cant fix it')

                    # anf.print_array_nicely(self._noise_matrices_dictionary[cluster_key]['averaged'])
                return {'averaged': self._noise_matrices_dictionary[cluster_key]['averaged']}
            else:
                return self._compute_noise_matrix_dependent(qubits_of_interest,
                                                            neighbors_of_interest)

        else:
            neighbors_key = 'q' + 'q'.join([str(s) for s in neighbors_of_interest])

            if neighbors_key in self._noise_matrices_dictionary[cluster_key]:
                return self._noise_matrices_dictionary[cluster_key][neighbors_key]
            else:
                return self._compute_noise_matrix_dependent(qubits_of_interest,
                                                            neighbors_of_interest)

    def compute_subset_noise_matrices_averaged(self,
                                               subsets_list: List[List[int]],
                                               show_progress_bar: Optional[bool] = False) -> None:
        """Description:
        computes averaged (over all other qubits) noise matrices on subsets_list of qubits

        :param subsets_list: subsets_list of qubit indices
        :param show_progress_bar: whether to show animated progress bar. requires tqdm package

        """
        # self.normalize_marginals()
        subsets_range = range(len(subsets_list))
        if show_progress_bar:
            from tqdm import tqdm
            subsets_range = tqdm(subsets_range)

        for subset_index in subsets_range:
            self._compute_noise_matrix_averaged(subsets_list[subset_index])
