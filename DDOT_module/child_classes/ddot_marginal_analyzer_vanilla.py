"""
Created on 28.04.2021

@author: Filip Maciejewski
@contact: filip.b.maciejewski@gmail.com
"""

import copy
import numpy as np
from typing import Optional
from collections import defaultdict
from QREM.DDOT_module.parent_classes.marginals_analyzer_base import MarginalsAnalyzerBase
from QREM import ancillary_functions as anf
from typing import Optional, List


class DDOTMarginalsAnalyzer(MarginalsAnalyzerBase):
    """
        1
    """

    def __init__(self,
                 results_dictionary_ddot: dict,
                 reverse_counts: bool,
                 marginals_dictionary: Optional[dict] = None,
                 noise_matrices_dictionary: Optional[dict] = None
                 ) -> None:

        super().__init__(results_dictionary_ddot,
                         reverse_counts,
                         marginals_dictionary
                         )

        if noise_matrices_dictionary is None:
            noise_matrices_dictionary = {}

        self._noise_matrices_dictionary = noise_matrices_dictionary

    @property
    def noise_matrices_dictionary(self) -> dict:
        return self._noise_matrices_dictionary

    @noise_matrices_dictionary.setter
    def noise_matrices_dictionary(self,
                                  noise_matrices_dictionary: dict) -> None:
        self._noise_matrices_dictionary = noise_matrices_dictionary

    @staticmethod
    def get_qubits_key(list_of_qubits: List[int]):
        return 'q' + 'q'.join([str(s) for s in list_of_qubits])

    @staticmethod
    def get_noise_matrix_from_counts_dict(results_dictionary: dict) -> np.ndarray:
        """Return noise matrix from counts dictionary.
        Assuming that the results are given only for qubits of interest.
        :param results_dictionary: results dictionary for which KEY is the bitstring denoting INPUT CLASSICAL STATE, while VALUE is the probability vector of results

        :return: noise_matrix: the array representing noise on qubits on which the experiments were performed
        """

        number_of_qubits = len(list(results_dictionary.keys())[0])
        noise_matrix = np.zeros((2 ** number_of_qubits, 2 ** number_of_qubits))
        for input_state, probability_vector in results_dictionary.items():
            noise_matrix[:, int(input_state, 2)] = probability_vector[:, 0]
        return noise_matrix

    @staticmethod
    def average_noise_matrices_over_some_qubits(matrices_cluster: dict,
                                                all_neighbors: list,
                                                qubits_to_be_left: list) -> dict:
        """
          Given dictionary of noise matrices, average them over some qubits.

         :param matrices_cluster: dictionary for which KEY is classical INPUT state of neighbors,
                                  and VALUE is a noise matrix
         :param all_neighbors: list of neighbors of given cluster
         :param qubits_to_be_left: qubits which we are interested in and we do not average over them

         :return: dictionary of noise matrices
                    depending on the state of neighbors MINUS qubits_to_be_averaged_over

         """

        # enumerated_neighbors = dict(enumerate(all_neighbors))
        rev_map = anf.get_enumerated_rev_map_from_indices(all_neighbors)
        averaging_terms_i = int(2 ** (len(all_neighbors) - len(qubits_to_be_left)))

        states_after_averaging = anf.register_names_qubits(range(len(qubits_to_be_left)),
                                                           len(qubits_to_be_left), False)
        averaged_dimension = list(matrices_cluster.values())[0].shape[0]

        averaged_matrices_cluster = {
            state: np.zeros((averaged_dimension, averaged_dimension), dtype=float) for state in
            states_after_averaging}

        # averaged_matrices_cluster['averaged'] = np.zeros((averaged_dimension, averaged_dimension), dtype=float)

        qubits_to_be_averaged_over = list(set(all_neighbors).difference(set(qubits_to_be_left)))
        qubits_to_be_averaged_over_mapped = [rev_map[q] for q in qubits_to_be_averaged_over]

        # print(matrices_cluster)

        # print(all_neighbors,qubits_to_be_left)
        for neighbors_state, conditional_noise_matrix in matrices_cluster.items():
            list_string_i = list(copy.deepcopy(neighbors_state))

            list_string_i = list(np.delete(list_string_i, qubits_to_be_averaged_over_mapped))

            string_joined_i = ''.join(list_string_i)

            # print(neighbors_state,string_joined_i,conditional_noise_matrix)
            # raise KeyError
            # if string_joined_i == '' or string_joined_i:
            #     try:
            #         averaged_matrices_cluster['averaged'] += conditional_noise_matrix / averaging_terms_i
            #     except(KeyError):
            #         averaged_matrices_cluster['averaged'] = conditional_noise_matrix/averaging_terms_i
            # else:
            averaged_matrices_cluster[string_joined_i] += conditional_noise_matrix / averaging_terms_i

        # print(averaged_matrices_cluster)
        return averaged_matrices_cluster

    def _compute_noise_matrix_averaged(self,
                                       subset: list) -> np.ndarray:
        '''Noise matrix for subset of qubits, averaged over all other qubits
           Defaultly takes data from self._marginals_dictionary. If data is not present, then it
           calculates marginals for given subset
           and updates the class's property self.marginals_dictionary

        '''

        # TODO: Perhaps add possiblity of using existing marginals for bigger subset that includes
        #      target subset

        subset_key = 'q' + 'q'.join([str(s) for s in subset])
        marginals_dictionary_ddot = self._marginals_dictionary

        # TODO: testing some issue with normalization
        testing_method = False

        # print(marginals_dictionary_ddot['q0q1'])

        marginal_dict_now = defaultdict(float)
        for what_we_put, dictionary_marginals_now in marginals_dictionary_ddot.items():
            input_marginal = ''.join([what_we_put[x] for x in subset])

            # print(what_we_put,subset_key)

            if subset_key not in dictionary_marginals_now.keys():
                self.compute_marginals(what_we_put, [subset])

            marginal_dict_now[input_marginal] += dictionary_marginals_now[subset_key]

            # print(subset_key, marginal_dict_now)

            if testing_method:
                for key_small in marginal_dict_now.keys():
                    marginal_dict_now[key_small] *= 1 / np.sum(marginal_dict_now[key_small])

        if not testing_method:
            for key_small in marginal_dict_now.keys():
                marginal_dict_now[key_small] *= 1 / np.sum(marginal_dict_now[key_small])

        noise_matrix_averaged = self.get_noise_matrix_from_counts_dict(marginal_dict_now)
        self._noise_matrices_dictionary[subset_key] = {'averaged': noise_matrix_averaged}

        return noise_matrix_averaged

    def _get_noise_matrix_averaged(self,
                                   subset: List[int]) -> np.ndarray:
        subset_key = 'q' + 'q'.join([str(s) for s in subset])
        try:
            return self._noise_matrices_dictionary[subset_key]['averaged']
        except(KeyError):
            return self._compute_noise_matrix_averaged(subset)

    def _compute_noise_matrix_dependent(self,
                                        qubits_of_interest: List[int],
                                        neighbors_of_interest: List[int]) -> dict:
        """Return lower-dimensional effective noise matrices acting on "qubits_of_interest" conditioned on input states of "all_neighbors of interest"
            :param qubits_of_interest (list of ints): list of integers labeling the qubits in marginal we are interested in
            :param neighbors_of_interest (list of ints): list of integers labeling the qubits that affet noise matrix on "qubits_of_interest"

            :return: lambdas_dict (dictionary):
                     dictionary with two KEYS:
                     lambdas_dict['lambdas'] = DICTIONARY, where KEY is the input state of "neighbors_of_interest", and VALUE is the noise matrix acting on "qubits_of_interested" for this fixed input state of all_neighbors
                     lambdas_dict['all_neighbors'] = list of all_neighbors (this either copies param "neighbors_of_interest" or returns all_neighbors mapped according to "mapping" param)

        """

        # If there are no all_neighbors, then this corresponds to averaging over all qubits except "qubits_of_interest"
        if len(neighbors_of_interest) == 0 or neighbors_of_interest is None:
            cluster_string = self.get_qubits_key(qubits_of_interest)
            if 'averaged' in self._noise_matrices_dictionary.keys():
                return {'averaged': self._noise_matrices_dictionary[cluster_string]}
            else:
                noise_matrix = self._get_noise_matrix_averaged(qubits_of_interest)
                return {'averaged': noise_matrix}

        # check if there is no collision between qubits of interest and all_neighbors of interest (if there is, then the model won't be consistent)
        if len(anf.lists_intersection(qubits_of_interest, neighbors_of_interest)) != 0:
            print(qubits_of_interest, neighbors_of_interest)
            raise ValueError('Wrong indices')

        # first, get averaged noise matrix on qubits of interest and all_neighbors of interest
        # TODO: make sure that qubit indices are correct

        our_qubits = sorted(qubits_of_interest + neighbors_of_interest)
        our_qubits_enumerated = anf.get_enumerated_rev_map_from_indices(our_qubits)

        big_lambda = self._get_noise_matrix_averaged(our_qubits)
        # big_lambda = self.get_noise_matrix_averaged(qubits_of_interest + neighbors_of_interest)

        # What is the number of qubits in the whole space
        total_number_of_qubits = int(np.log2(big_lambda.shape[0]))

        # What is the number of qubits in marginal of interest
        number_of_qubits_of_interest = len(qubits_of_interest)

        # What is the number of all_neighbors
        number_of_neighbors = len(neighbors_of_interest)

        # Normalization when averaging over states of non-neighbours (each with the same probability)
        normalization = 2 ** (
                total_number_of_qubits - number_of_neighbors - number_of_qubits_of_interest)

        # classical register on all qubits
        classical_register_big = ["{0:b}".format(i).zfill(total_number_of_qubits) for i in
                                  range(2 ** total_number_of_qubits)]

        # classical register on neighbours
        classical_register_neighbours = ["{0:b}".format(i).zfill(number_of_neighbors) for i in
                                         range(2 ** number_of_neighbors)]

        # create dictionary of the marginal states of qubits of interest and all_neighbors for the whole
        # register (this function is storing data which could also be calculated in situ
        # in the loops later, but this is faster)

        indices_small = {}
        for s in classical_register_big:
            small_string = ''.join([list(s)[our_qubits_enumerated[b]] for b in qubits_of_interest])
            neighbours_string = ''.join(
                [list(s)[our_qubits_enumerated[b]] for b in neighbors_of_interest])
            # first place in list is label for state of "qubits_of_interest" and second for "neighbors_of_interest
            indices_small[s] = [small_string, neighbours_string]

        # initiate dictionary for which KEY is input state of all_neighbors and VALUE will the the corresponding noise matrix on "qubits_of_interest"
        small_lambdas = {
            s: np.zeros((2 ** number_of_qubits_of_interest, 2 ** number_of_qubits_of_interest)) for s
            in
            classical_register_neighbours}

        # go through all classical states
        for i in range(2 ** total_number_of_qubits):
            for j in range(2 ** total_number_of_qubits):
                lambda_element = big_lambda[i, j]

                # input state of all qubits in binary format
                input_state = "{0:b}".format(j).zfill(total_number_of_qubits)
                # measured state of all qubits in binary format
                measured_state = "{0:b}".format(i).zfill(total_number_of_qubits)

                # input state of "qubits_of_interest" in binary format
                input_state_small = indices_small[input_state][0]
                # measured state of "qubits_of_interest" in binary format
                measured_state_small = indices_small[measured_state][0]

                # input state of "neighbors_of_interest" in binary format
                input_state_neighbours = indices_small[input_state][1]

                # element of small lambda labeled by (measured state | input state), and the lambda itself is labeled by input state of all_neighbors
                small_lambdas[input_state_neighbours][
                    int(measured_state_small, 2), int(input_state_small, 2)] += lambda_element

        # normalize matrices
        for s in classical_register_neighbours:
            small_lambdas[s] *= 1 / normalization

        # small_lambdas['all_neighbors'] = neighbors_of_interest

        cluster_string = 'q' + 'q'.join(str(s) for s in qubits_of_interest)
        neighbours_string = 'q' + 'q'.join(str(s) for s in neighbors_of_interest)


        if cluster_string not in self._noise_matrices_dictionary.keys():
            # TODO: make sure if it is the same as averaging over all other qubits
            lam_av = np.zeros((2 ** number_of_qubits_of_interest, 2 ** number_of_qubits_of_interest))
            for s in small_lambdas.keys():
                lam_av += small_lambdas[s]
            lam_av *= 1 / 2 ** number_of_qubits_of_interest
            self._noise_matrices_dictionary[cluster_string] = {'averaged': lam_av}

        self._noise_matrices_dictionary[cluster_string][neighbours_string] = small_lambdas

        return self._noise_matrices_dictionary[cluster_string][neighbours_string]

    def _get_noise_matrix_dependent(self,
                                    qubits_of_interest: List[int],
                                    neighbors_of_interest: List[int]) -> dict:
        cluster_key = 'q' + 'q'.join([str(s) for s in qubits_of_interest])

        if len(neighbors_of_interest) == 0 or neighbors_of_interest is None:
            neighbors_key = 'averaged'

            if neighbors_key in self._noise_matrices_dictionary[cluster_key]:
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

    def compute_subset_noise_matrices(self,
                                      subsets_list: List[List[int]],
                                      show_progress_bar: Optional[bool] = False):

        if show_progress_bar:
            from tqdm import tqdm
            for subset_index in tqdm(range(len(subsets_list))):
                self._compute_noise_matrix_averaged(subsets_list[subset_index])
        else:
            for subset in subsets_list:
                self._compute_noise_matrix_averaged(subset)
