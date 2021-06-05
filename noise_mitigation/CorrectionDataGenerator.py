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

import numpy as np
from typing import Optional, Dict, Union, List
from noise_characterization.modeling.NoiseModelGenerator import NoiseModelGenerator
from noise_characterization.modeling.GlobalNoiseMatrixCreator import GlobalNoiseMatrixCreator
from functions import povmtools, ancillary_functions as anf


class CorrectionDataGenerator(NoiseModelGenerator):
    """
        Main class used to calculate data needed for noise-mitigation on marginals_dictionary, based on provided
        noise model.

        NOTE: Currently it handles properly only two-qubit marginals_dictionary (e.g., experiments involving
              estimation of 2-local Hamiltonians)

        The correction data consists of the following:
        - :param: 'correction_indices' - for each marginal of interest (e.g., 'q0q1'), specify label for
                                 marginal that needs to be corrected and then coarse-grained in order
                                 to perform noise-mitigation. For example, if q0 is strongly correlated
                                 with q5 (i.e., they are in the same cluster), it is advisable to first
                                 perform noise mitigation on marginal 'q0q1q5', and then coarse-grain
                                 it to obtain 'q0q1'.

                                 The format we use here is dictionary where KEY is label for marginal
                                 of interest, and VALUE is label for marginal that needs to be
                                 calculated first.
                                 So the entry for example above would look like:
                                 correction_indices['q0q1'] = 'q0q1q5'


        - :param: 'noise_matrices' - the noise matrices representing effective noise matrix acting on marginals_dictionary
                            specified by values of correction_indices dictionary

                            This is dictionary where KEY is subset label and VALUE is noise matrix


        - :param: 'correction_matrices' - inverses of noise_matrices, convention used is the same
    """

    # TODO FBM: generalize for more than two-qubit subsets_list
    # TODO FBM: add mitigation errors

    def __init__(self,
                 results_dictionary_ddt: Dict[str, Dict[str, int]],
                 bitstrings_right_to_left: bool,
                 number_of_qubits: int,
                 marginals_dictionary: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
                 clusters_list: Optional[List[List[int]]] = None,
                 neighborhoods: Optional[Dict[str, List[int]]] = None,
                 noise_matrices_dictionary: Optional[
                     Dict[str, Union[np.ndarray, Dict[str, Dict[str, np.ndarray]]]]] = None
                 ) -> None:

        super().__init__(results_dictionary_ddt,
                         bitstrings_right_to_left,
                         number_of_qubits,
                         marginals_dictionary,
                         noise_matrices_dictionary,
                         clusters_list,
                         neighborhoods
                         )
        self._number_of_qubits = number_of_qubits
        self._qubit_indices = range(number_of_qubits)

        if clusters_list is None:
            clusters_lists_dictionary, clusters_labels_dictionary = {}, {}

        else:
            clusters_lists_dictionary = {'q%s' % qi: [] for qi in self._qubit_indices}
            for cluster in self._clusters_list:
                for qi in cluster:
                    clusters_lists_dictionary['q%s' % qi] = cluster

            clusters_labels_dictionary = {
                'q%s' % qi: self.get_qubits_key(clusters_lists_dictionary['q%s' % qi]) for qi in
                self._qubit_indices}

        self._noise_matrices = {}
        self._correction_matrices = {}
        self._mitigation_errors = {}
        self._correction_indices = {}

        self._clusters_lists_dictionary = clusters_lists_dictionary
        self._clusters_labels_dictionary = clusters_labels_dictionary

    def _set_clusters_dictionary(self) -> None:
        """
        Function that updates class' properties needed (later) to calculate correction data
        """

        clusters_lists_dictionary = {'q%s' % qi: [] for qi in self._qubit_indices}
        for cluster in self._clusters_list:
            for qi in cluster:
                clusters_lists_dictionary['q%s' % qi] = cluster
        clusters_labels_dictionary = {
            'q%s' % qi: self.get_qubits_key(clusters_lists_dictionary['q%s' % qi]) for qi in
            self._qubit_indices}
        self._clusters_lists_dictionary = clusters_lists_dictionary
        self._clusters_labels_dictionary = clusters_labels_dictionary

    def _construct_local_full_noise_model(self,
                                          qubit_i: int,
                                          qubit_j: int,
                                          intersection_i: Optional[List[int]] = None,
                                          intersection_j: Optional[List[int]] = None
                                          ) -> np.ndarray:
        """
        This is function to obtain local noise model for two clusters to which both qubits belong.
        It is slightly tedious, the main issue here is to properly sort qubits and format input data
        for GlobalNoiseMatrixCreator (different class)

        :param qubit_i: qubit that belongs to cluster i
        :param qubit_j: as above
        :param intersection_i: qubits that are neighbors of cluster i and belong to cluster j
        :param intersection_j: as above
        :return:
        """

        # get needed information from class' properties
        cluster_i = self._clusters_lists_dictionary['q%s' % qubit_i]
        cluster_j = self._clusters_lists_dictionary['q%s' % qubit_j]

        string_cluster_i = self._clusters_labels_dictionary['q%s' % qubit_i]
        string_cluster_j = self._clusters_labels_dictionary['q%s' % qubit_j]

        neighbors_i = self._neighborhoods[string_cluster_i]
        neighbors_j = self._neighborhoods[string_cluster_j]

        # if not provided, calculate it
        if intersection_i is None:
            intersection_i = anf.lists_intersection(neighbors_i, cluster_j)
        if intersection_j is None:
            intersection_j = anf.lists_intersection(neighbors_j, cluster_i)

        # Take cluster noise matrices depending on neighbors states
        matrices_cluster_i = self.get_noise_matrix_dependent(cluster_i,
                                                             neighbors_i)
        matrices_cluster_j = self.get_noise_matrix_dependent(cluster_j,
                                                             neighbors_j)

        # average over neighbours of first cluster
        # that do not include the members of second cluster
        averaged_matrices_cluster_i = self.average_noise_matrices_over_some_qubits(
            matrices_cluster_i,
            neighbors_i,
            intersection_i
        )
        # and vice versa
        averaged_matrices_cluster_j = self.average_noise_matrices_over_some_qubits(
            matrices_cluster_j,
            neighbors_j,
            intersection_j)

        """
        Here we will need to properly relabel qubits because noise matrix creator requires 
        numbering them from 0 to number of qubits.
        """
        # Label qubits from 0
        enumerated_map_reversed = anf.get_reversed_enumerated_from_indices(
            cluster_i + cluster_j)

        qubit_indices_enumerated = []
        for cluster_now in [cluster_i, cluster_j]:
            qubit_indices_enumerated.append(
                [enumerated_map_reversed[ci] for ci in cluster_now])

        cluster_key_for_construction_i = self.get_qubits_key(
            qubit_indices_enumerated[0])
        cluster_key_for_construction_j = self.get_qubits_key(
            qubit_indices_enumerated[1])

        # this is format of noise matrices dictionary accepted by GlobalNoiseMatrixCreator
        properly_formatted_lambdas = {cluster_key_for_construction_i: averaged_matrices_cluster_i,
                                      cluster_key_for_construction_j: averaged_matrices_cluster_j
                                      }

        # this is format for neighbors dictionary accepted by GlobalNoiseMatrixCreator
        neighbors_for_construction = {
            cluster_key_for_construction_i: [enumerated_map_reversed[ci] for ci in
                                             intersection_i],
            cluster_key_for_construction_j: [enumerated_map_reversed[cj] for cj in
                                             intersection_j]}

        # get instance of GlobalNoiseMatrixCreator
        big_lambda_creator_now = GlobalNoiseMatrixCreator(
            properly_formatted_lambdas,
            qubit_indices_enumerated,
            neighbors_for_construction)

        # get noise matrix
        local_noise_matrix = \
            big_lambda_creator_now.compute_global_noise_matrix()

        return local_noise_matrix

    @staticmethod
    def _sort_qubits_matrix(
            local_noise_matrix: np.ndarray,
            cluster_i: List[int],
            cluster_j: List[int]) -> np.ndarray:
        """
        :param local_noise_matrix: noise matrix acting on cluster_i and cluster_j
        :param cluster_i: list of qubits' indices
        :param cluster_j: list of qubits' indices
        :return: permuted local_noise_matrix
        """

        all_qubits_list = cluster_i + cluster_j

        # This is our target
        sorted_qubits_dictionary = anf.enumerated_dictionary(sorted(all_qubits_list))
        # This is what we have
        qubits_dictionary = anf.enumerated_dictionary(all_qubits_list)

        # While what we have is not equal to target we sort qubits.
        # The following loop performs series of SWAPs to properly order qubits.
        while qubits_dictionary != sorted_qubits_dictionary:
            for index_qubit_here in range(len(all_qubits_list) - 1):
                if all_qubits_list[index_qubit_here + 1] < all_qubits_list[index_qubit_here]:
                    # if two qubits are not sorted in ascending order, we permute matrix
                    # this corresponds to exchanging place of two qubits in the Hilbert space
                    local_noise_matrix = povmtools.permute_matrix(local_noise_matrix,
                                                                  len(all_qubits_list),
                                                                  [index_qubit_here + 1,
                                                                   index_qubit_here + 2])
                    # update indices to keep track of already made swaps
                    all_qubits_list[index_qubit_here], all_qubits_list[
                        index_qubit_here + 1] = all_qubits_list[index_qubit_here + 1], all_qubits_list[
                        index_qubit_here]
            # update whole dictionary
            qubits_dictionary = anf.enumerated_dictionary(all_qubits_list)
        return local_noise_matrix

    def _compute_pair_correction_data(self,
                                      pair: List[int]) -> None:
        """
        For given pair of qubits, get correction data required
        to correct corresponding two-qubit marginal.
        NOTE: see class' description

        :param pair: list of qubit indices
        """

        qubit_i, qubit_j = pair[0], pair[1]

        cluster_i = self._clusters_lists_dictionary['q%s' % qubit_i]
        cluster_j = self._clusters_lists_dictionary['q%s' % qubit_j]

        if cluster_i == cluster_j:
            # Check if qubits are in the same cluster. If yes, we just take average
            # noise matrix on that cluster.
            averaged_matrix_clusters_i_j = self._get_noise_matrix_averaged(
                sorted(anf.lists_sum(cluster_i, cluster_j)))

        else:
            string_cluster_i = self._clusters_labels_dictionary['q%s' % qubit_i]
            string_cluster_j = self._clusters_labels_dictionary['q%s' % qubit_j]
            # If qubits are in different clusters, we have two options to consider below...
            intersection_i, intersection_j = \
                anf.lists_intersection(
                    self._neighborhoods[string_cluster_i],
                    cluster_j), \
                anf.lists_intersection(
                    self._neighborhoods[string_cluster_j],
                    cluster_i)

            if len(intersection_i) == 0 and len(intersection_j) == 0:
                # Check if clusters contain each others neighbors.
                # If not, the noise matrix is simply potentially_stochastic_matrix
                # tensor product of cluster matrices
                averaged_matrix_cluster_i = self._get_noise_matrix_averaged(cluster_i)
                averaged_matrix_cluster_j = self._get_noise_matrix_averaged(cluster_j)

                averaged_matrix_clusters_i_j = np.kron(averaged_matrix_cluster_i,
                                                       averaged_matrix_cluster_j)

            else:
                # Check if clusters are each others neighbors.
                # If yes, the noise matrix needs to be constructed using
                # cluster-neighborhoods noise model with treating
                # some members of clusters as neighbours
                averaged_matrix_clusters_i_j = self._construct_local_full_noise_model(
                    qubit_i=qubit_i,
                    qubit_j=qubit_j,
                    intersection_i=intersection_i,
                    intersection_j=intersection_j)

        local_noise_matrix = averaged_matrix_clusters_i_j
        # check whether qubits are properly sorted
        if cluster_i != cluster_j and cluster_i + cluster_j != sorted(cluster_i + cluster_j):
            # if qubits are not sorted, noise matrix needs to be permuted
            local_noise_matrix = self._sort_qubits_matrix(
                local_noise_matrix=averaged_matrix_clusters_i_j,
                cluster_i=cluster_i,
                cluster_j=cluster_j)

        whole_marginal = sorted(anf.lists_sum(cluster_i, cluster_j))
        correction_matrix = np.linalg.inv(local_noise_matrix)

        string_marginal = ''.join(['q%s' % q_index for q_index in whole_marginal])
        self._correction_indices['q%s' % qubit_i] = string_marginal
        self._correction_indices['q%s' % qubit_j] = string_marginal
        self._correction_indices['q%sq%s' % (qubit_i, qubit_j)] = string_marginal

        self._noise_matrices[string_marginal] = local_noise_matrix
        self._correction_matrices[string_marginal] = correction_matrix

    def get_pairs_correction_data(self,
                                  pairs_list: List[List[int]],
                                  show_progress_bar: Optional[bool] = False) -> dict:
        """
        For pairs of qubits in the list, get correction data required
        to correct corresponding two-qubit marginals_dictionary.
        NOTE: see class' description for details

        :param pairs_list:
        :param show_progress_bar:
        :return: correction_data_dictionary
        """

        # TODO FBM: possibly change resetting
        self._set_clusters_dictionary()
        self._noise_matrices = {}
        self._correction_matrices = {}
        self._mitigation_errors = {}
        self._correction_indices = {}

        range_pairs = range(len(pairs_list))

        if show_progress_bar:
            from tqdm import tqdm
            range_pairs = tqdm(range_pairs)

        for pair_index in range_pairs:
            pair = pairs_list[pair_index]
            pair_key = self.get_qubits_key(pair)
            if pair_key not in self._correction_indices.keys():
                self._compute_pair_correction_data(pair)

        return {'correction_matrices': self._correction_matrices,
                'noise_matrices': self._noise_matrices,
                'correction_indices': self._correction_indices}
