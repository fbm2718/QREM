"""
Created on 29.04.2021


@authors: Filip Maciejewski, Oskar Słowik
@contact: filip.b.maciejewski@gmail.com

REFERENCES:
[0] Filip B. Maciejewski, Zoltán Zimborás, Michał Oszmaniec,
"Mitigation of readout noise in near-term quantum devices
by classical post-processing based on detector tomography",
Quantum 4, 257 (2020)

[0.5] Filip B. Maciejewski, Flavio Baccari Zoltán Zimborás, Michał Oszmaniec,
"Modeling and mitigation of realistic readout noise
with applications to the Quantum Approximate Optimization Algorithm",
arxiv: arXiv:2101.02331 (2021)

"""

import numpy as np
from typing import Optional, Dict, Union, List
from QREM import ancillary_functions as anf
from DDOT_module.child_classes.noise_model_generator_vanilla import NoiseModelGenerator
from DDOT_module.child_classes.global_noise_matrix_creator import GlobalNoiseMatrixCreator


class CorrectionDataGenerator(NoiseModelGenerator):
    """
        Main class used to calculate data needed for noise-mitigation on marginals, based on provided
        noise model.

        NOTE: Currently it handles properly only two-qubit marginals (e.g., experiments involving
              estimation of 2-local Hamiltonians)

        The correction data consists of the following:
        - 'correction_indices' - for each marginal of interest (e.g., 'q0q1'), specify label for
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


        - 'noise_matrices' - the noise matrices representing effective noise matrix acting on marginals
                            specified by values of correction_indices dictionary

                            This is dictionary where KEY is subset label and VALUE is noise matrix


        - 'correction_matrices' - inverses of noise_matrices, convention used is the same
    """

    #TODO FBM: finish this documentation

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
            clusters_dictionary = {}

        else:
            clusters_dictionary = {'q%s' % qi: [] for qi in self._qubit_indices}
            for cluster in self._clusters_list:
                for qi in cluster:
                    for qj in cluster:
                        if qi != qj:
                            clusters_dictionary['q%s' % qi].append(qj)

        self._noise_matrices = {}
        self._correction_matrices = {}
        self._mitigation_errors = {}
        self._correction_indices = {}

        self._clusters_dictionary = clusters_dictionary

    # TODO FBM: add higher locality of hamiltonians

    def set_clusters_dictionary(self):
        clusters_dictionary = {'q%s' % qi: [] for qi in self._qubit_indices}
        for cluster in self._clusters_list:
            for qi in cluster:
                for qj in cluster:
                    if qi != qj:
                        clusters_dictionary['q%s' % qi].append(qj)
        self._clusters_dictionary = clusters_dictionary

    def compute_pairs_correction_matrices(self,
                                          pairs_list: List[List[int]]) -> None:
        """
        :param pairs_list:
        :return:
        """

        # TODO FBM: split this into several smaller functions

        # TODO FBM: generalize for more than two-qubit subsets

        # TODO FBM: add mitigation errors

        self.set_clusters_dictionary()

        testing_averaging = True

        for pair_index in range(len(pairs_list)):
            pair = pairs_list[pair_index]
            i, j = pair[0], pair[1]
            string_pair = 'q%sq%s' % (i, j)

            cluster_i = sorted([i] + self._clusters_dictionary['q%s' % i])
            cluster_j = sorted([j] + self._clusters_dictionary['q%s' % j])

            # print(cluster_i,cluster_j)

            string_cluster_i = ''.join(['q%s' % qi for qi in cluster_i])
            string_cluster_j = ''.join(['q%s' % qi for qi in cluster_j])

            neighborhoods_cluster_i = self._neighborhoods[string_cluster_i]
            neighborhoods_cluster_j = self._neighborhoods[string_cluster_j]

            if len(anf.lists_intersection(cluster_i, cluster_j)) != 0:
                # Check if clusters overlap. If yes, we treat them as single, big cluster
                # and construct cluster-neighborhood noise model
                dependencies_clusters_i_j = sorted(
                    anf.lists_sum(neighborhoods_cluster_i,
                                  neighborhoods_cluster_j))

                if testing_averaging:
                    averaged_matrix_clusters_i_j = self._get_noise_matrix_averaged(
                        sorted(anf.lists_sum(cluster_i, cluster_j)))

                else:
                    matrices_clusters = self.get_noise_matrix_dependent(
                        anf.lists_sum(cluster_i, cluster_j),
                        dependencies_clusters_i_j)

                    averaged_matrix_clusters_i_j = sum(
                        [lam for lam in matrices_clusters.values()]) / 2 ** (
                                                       len(dependencies_clusters_i_j))
            else:
                # Check if clusters overlap. If not, we treat them as separate clusters.
                dependencies_cluster_i = sorted(self._neighborhoods[string_cluster_i])
                dependencies_cluster_j = sorted(self._neighborhoods[string_cluster_j])

                matrices_cluster_i = self.get_noise_matrix_dependent(cluster_i,
                                                                     neighborhoods_cluster_i)
                # print('\n', cluster_j, neighborhoods_cluster_j)
                matrices_cluster_j = self.get_noise_matrix_dependent(cluster_j,
                                                                     neighborhoods_cluster_j)

                intersection_i, intersection_j = anf.lists_intersection(dependencies_cluster_i,
                                                                        cluster_j), \
                                                 anf.lists_intersection(dependencies_cluster_j,
                                                                        cluster_i)

                if len(intersection_i) == 0 and len(intersection_j) == 0:
                    # Check if clusters contain each others neighbors.
                    # If not, the noise matrix is simply a tensor product of clusters.

                    if testing_averaging:
                        averaged_matrix_cluster_i = self._get_noise_matrix_averaged(cluster_i)
                        averaged_matrix_cluster_j = self._get_noise_matrix_averaged(cluster_j)

                        # np.testing.assert_array_almost_equal(averaged_matrix_cluster_i,sum(
                        #     [lam_i for lam_i in matrices_cluster_i.values()]) / 2 ** (
                        #                                 len(dependencies_cluster_i)))

                    else:
                        averaged_matrix_cluster_i = sum(
                            [lam_i for lam_i in matrices_cluster_i.values()]) / 2 ** (
                                                        len(dependencies_cluster_i))
                        averaged_matrix_cluster_j = sum(
                            [lam_j for lam_j in matrices_cluster_j.values()]) / 2 ** (
                                                        len(dependencies_cluster_j))

                    averaged_matrix_clusters_i_j = np.kron(averaged_matrix_cluster_i,
                                                           averaged_matrix_cluster_j)

                else:
                    # Check if clusters are each others neighbors.
                    # If yes, the noise matrix needs to be constructed using
                    # cluster-neighborhoods noise model with treating
                    # some members of clusters as neighbours

                    # average over neighbours of first cluster
                    # that do not include the members of second cluster
                    if len(dependencies_cluster_i) != 0:
                        averaged_matrices_cluster_i = self.average_noise_matrices_over_some_qubits(
                            matrices_cluster_i,
                            dependencies_cluster_i,
                            intersection_i
                        )
                    else:
                        averaged_matrices_cluster_i = matrices_cluster_i

                    # average over neighbours of second cluster
                    # that do not include the members of first cluster
                    if len(dependencies_cluster_j) != 0:
                        averaged_matrices_cluster_j = self.average_noise_matrices_over_some_qubits(
                            matrices_cluster_j,
                            dependencies_cluster_j,
                            intersection_j)
                    else:
                        averaged_matrices_cluster_j = matrices_cluster_j

                    # Sort indices
                    rev_map_enumerated = anf.get_reversed_enumerated_from_indices(
                        cluster_i + cluster_j)

                    qubit_indices_for_construction = []
                    for cluster_now in [cluster_i, cluster_j]:
                        qubit_indices_for_construction.append(
                            [rev_map_enumerated[ci] for ci in cluster_now])

                    properly_formatted_lambdas = {
                        self.get_qubits_key(
                            qubit_indices_for_construction[0]): averaged_matrices_cluster_i,
                        self.get_qubits_key(
                            qubit_indices_for_construction[1]): averaged_matrices_cluster_j
                    }

                    neighbors_for_construction = [[rev_map_enumerated[ci] for ci in
                                                   intersection_i],
                                                  [rev_map_enumerated[cj] for cj in
                                                   intersection_j]]

                    neighbors_for_construction_modified = []
                    for index_subset in range(len(neighbors_for_construction)):
                        subset_neighbors = neighbors_for_construction[index_subset]

                        if len(subset_neighbors) == 0:
                            neighbors_for_construction_modified.append(None)
                        else:
                            neighbors_for_construction_modified.append(subset_neighbors)

                    neighbors_for_construction = neighbors_for_construction_modified

                    big_lambda_creator_now = GlobalNoiseMatrixCreator(
                        properly_formatted_lambdas)
                    averaged_matrix_clusters_i_j = \
                        big_lambda_creator_now.compute_global_noise_matrix(
                            qubit_indices_for_construction,
                            neighbors_for_construction
                        )

            # check whether qubits are properly sorted
            sorted_quest = True
            if cluster_i != cluster_j:
                for ccc1 in cluster_j:
                    for ccc0 in cluster_i:
                        if ccc0 > ccc1:
                            sorted_quest = False

            whole_marginal = sorted(anf.lists_sum(cluster_i, cluster_j))
            if not sorted_quest:
                # TODO: swapping qubits, does it make sense?
                averaged_matrix = averaged_matrix_clusters_i_j
                import QREM.povmtools as qrem_pt

                qubits_in_here = cluster_i + cluster_j
                sorted_qubits_in_here = dict(enumerate(sorted(qubits_in_here)))

                # TODO FBM: why is it here?
                rev_map = anf.get_reversed_enumerated_from_indices(cluster_i + cluster_j)

                qubits_in_here_dict = dict(enumerate(qubits_in_here))

                while qubits_in_here_dict != sorted_qubits_in_here:
                    for index_qubit_here in range(len(qubits_in_here) - 1):
                        if qubits_in_here[index_qubit_here] < qubits_in_here[index_qubit_here + 1]:
                            pass
                        elif qubits_in_here[index_qubit_here + 1] < qubits_in_here[index_qubit_here]:
                            averaged_matrix = qrem_pt.permute_matrix(averaged_matrix,
                                                                     len(whole_marginal),
                                                                     [index_qubit_here + 1,
                                                                      index_qubit_here + 2])

                            anf.cool_print('Swapping qubits:', )
                            print(qubits_in_here[index_qubit_here],
                                  qubits_in_here[index_qubit_here + 1], index_qubit_here,
                                  index_qubit_here + 1)

                            qubits_in_here[index_qubit_here], qubits_in_here[
                                index_qubit_here + 1] = qubits_in_here[index_qubit_here + 1], \
                                                        qubits_in_here[index_qubit_here]

                    qubits_in_here_dict = dict(enumerate(qubits_in_here))

            else:
                averaged_matrix = averaged_matrix_clusters_i_j

            correction_matrix = np.linalg.inv(averaged_matrix)

            string_marginal = ''.join(['q%s' % qqq for qqq in whole_marginal])
            self._correction_indices[string_pair] = string_marginal
            self._correction_indices['q%s' % i] = string_marginal
            self._correction_indices['q%s' % j] = string_marginal
            self._noise_matrices[string_marginal] = averaged_matrix
            self._correction_matrices[string_marginal] = correction_matrix

    def get_pairs_correction_data(self,
                                  pairs_list: List[List[int]],
                                  show_progress_bar: Optional[bool] = False):

        # TODO FBM: possibly change resetting
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
                self.compute_pairs_correction_matrices([pair])

        return {'correction_matrices': self._correction_matrices,
                           'noise_matrices': self._noise_matrices,
                           'correction_indices': self._correction_indices}
