"""
Created on 29.04.2021

@author: Filip Maciejewski
@contact: filip.b.maciejewski@gmail.com
"""

import copy
import numpy as np
from typing import Optional, Dict
from tqdm import tqdm
from QREM import ancillary_functions as anf
from ..child_classes.ddot_marginal_analyzer_vanilla import DDOTMarginalsAnalyzer
from DDOT_module.child_classes.global_noise_matrix_creator import GlobalNoiseMatrixCreator


class CorrectionDataGenerator(DDOTMarginalsAnalyzer):
    """
        1
    """

    def __init__(self,
                 results_dictionary_ddot: dict,
                 reverse_counts: bool,
                 number_of_qubits: int,
                 marginals_dictionary: dict,
                 clusters_list: list,
                 neighborhoods: dict,
                 noise_matrices_dictionary: Optional[dict] = None
                 ) -> None:

        super().__init__(results_dictionary_ddot,
                         reverse_counts,
                         marginals_dictionary,
                         noise_matrices_dictionary
                         )
        self._number_of_qubits = number_of_qubits
        self._qubit_indices = range(number_of_qubits)

        self._clusters_list = clusters_list
        self._neighborhoods = neighborhoods

        clusters_dictionary = {'q%s' % qi: [] for qi in self._qubit_indices}
        for cluster in clusters_list:
            for qi in cluster:
                for qj in cluster:
                    if qi != qj:
                        clusters_dictionary['q%s' % qi].append(qj)

        self._clusters_dictionary = clusters_dictionary

        self._noise_matrices = {}
        self._correction_matrices = {}
        self._mitigation_errors = {}
        self._correction_indices = {}

    def compute_pairs_correction_matrices(self,
                                          pairs_list: list) -> None:

        # TODO FBM: add mitigation errors
        calculate_mitigation_errors = False

        highly_correlated_qubits = []

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
                    anf.lists_sum(neighborhoods_cluster_i, neighborhoods_cluster_j))

                matrices_clusters = self._get_noise_matrix_dependent(
                    anf.lists_sum(cluster_i, cluster_j),
                    dependencies_clusters_i_j)

                averaged_matrix_clusters_i_j = sum(
                    [lam for lam in matrices_clusters.values()]) / 2 ** (
                                                   len(dependencies_clusters_i_j))

                if calculate_mitigation_errors:
                    mitigation_errors_ij = 0
                    correction_norm_ij = np.linalg.norm(
                        np.linalg.inv(averaged_matrix_clusters_i_j), ord=1)
                    for lam in matrices_clusters.values():
                        err_now = np.linalg.norm(averaged_matrix_clusters_i_j - lam, ord=1)
                        if err_now > mitigation_errors_ij:
                            mitigation_errors_ij = err_now
                    mitigation_errors_ij *= correction_norm_ij


            else:
                # Check if clusters overlap. If not, we treat them as separate clusters.
                dependencies_cluster_i = sorted(self._neighborhoods[string_cluster_i])
                dependencies_cluster_j = sorted(self._neighborhoods[string_cluster_j])

                matrices_cluster_i = self._get_noise_matrix_dependent(cluster_i,
                                                                      neighborhoods_cluster_i)
                # print('\n', cluster_j, neighborhoods_cluster_j)
                matrices_cluster_j = self._get_noise_matrix_dependent(cluster_j,
                                                                      neighborhoods_cluster_j)

                intersection_i, intersection_j = anf.lists_intersection(dependencies_cluster_i,
                                                                        cluster_j), \
                                                 anf.lists_intersection(dependencies_cluster_j,
                                                                        cluster_i)

                if len(intersection_i) == 0 and len(intersection_j) == 0:

                    # Check if clusters contain each others all_neighbors.
                    # If not, the noise matrix is simply a tensor product of clusters.
                    averaged_matrix_cluster_i = sum(
                        [lam_i for lam_i in matrices_cluster_i.values()]) / 2 ** (
                                                    len(dependencies_cluster_i))

                    averaged_matrix_cluster_j = sum(
                        [lam_j for lam_j in matrices_cluster_j.values()]) / 2 ** (
                                                    len(dependencies_cluster_j))

                    averaged_matrix_clusters_i_j = np.kron(averaged_matrix_cluster_i,
                                                           averaged_matrix_cluster_j)

                    if calculate_mitigation_errors:
                        mitigation_errors_i, mitigation_errors_j = 0., 0.

                        correction_norm_i, correction_norm_j = np.linalg.norm(
                            np.linalg.inv(averaged_matrix_cluster_i), ord=1), \
                                                               np.linalg.norm(np.linalg.inv(
                                                                   averaged_matrix_cluster_j),
                                                                   ord=1)

                        for lam in matrices_cluster_i.values():
                            err_now = np.linalg.norm(averaged_matrix_cluster_i - lam, ord=1)

                            if err_now > mitigation_errors_i:
                                mitigation_errors_i = err_now

                        for lam in matrices_cluster_j.values():
                            err_now = np.linalg.norm(averaged_matrix_cluster_j - lam, ord=1)

                            if err_now > mitigation_errors_j:
                                mitigation_errors_j = err_now

                        if mitigation_errors_i > 0 and mitigation_errors_j > 0:
                            mitigation_errors_ij = mitigation_errors_i * mitigation_errors_j * correction_norm_i * correction_norm_j
                        elif mitigation_errors_i > 0 and mitigation_errors_j == 0:
                            mitigation_errors_ij = mitigation_errors_i * correction_norm_i
                        elif mitigation_errors_i == 0 and mitigation_errors_j > 0:
                            mitigation_errors_ij = mitigation_errors_j * correction_norm_j
                        else:
                            mitigation_errors_ij = 0

                else:
                    # Check if clusters are each others all_neighbors.
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
                    rev_map_enumerated = anf.get_enumerated_rev_map_from_indices(
                        cluster_i + cluster_j)

                    qubit_indices_for_construction = []
                    for clust in [cluster_i, cluster_j]:
                        qubit_indices_for_construction.append(
                            [rev_map_enumerated[ci] for ci in clust])

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
                        noise_matrices_dictionary=properly_formatted_lambdas)
                    averaged_matrix_clusters_i_j = \
                        big_lambda_creator_now.compute_global_noise_matrix(
                            qubit_indices_for_construction,
                            neighbors_for_construction
                        )

                    if calculate_mitigation_errors:
                        # TODO: FIX THIS PART
                        # qubits_clusters_ij = anf.get_enumerated_rev_map_from_indices(anf.lists_sum_multi([cluster_i,cluster_j]))
                        qubits_ij = anf.get_enumerated_rev_map_from_indices(anf.lists_sum_multi(
                            [cluster_i, cluster_j, dependencies_cluster_i,
                             dependencies_cluster_j]))

                        clu_i, clu_j = [qubits_ij[qc] for qc in cluster_i], [qubits_ij[qc] for qc
                                                                             in cluster_j],
                        deps_i, deps_j = [qubits_ij[qc] for qc in dependencies_cluster_i], [
                            qubits_ij[qc] for qc in dependencies_cluster_j]

                        all_qubit_ij = anf.lists_sum_multi([deps_i, deps_j])
                        all_qubits_outside_ij = list(
                            set(all_qubit_ij).difference(set(clu_i + clu_j)))

                        possible_states_ij_outside = anf.register_names_qubits(
                            range(len(all_qubits_outside_ij)), len(all_qubits_outside_ij))

                        possible_states_ij = []
                        for k in range(len(possible_states_ij_outside)):
                            state_here = possible_states_ij_outside[k]
                            better_statesize = len(list(state_here)) + len(clu_i) + len(clu_j)
                            new_state = np.zeros((better_statesize), dtype=str)

                            for ciiiii in clu_i:
                                new_state[ciiiii] = '0'
                            for cjjjjj in clu_j:
                                new_state[cjjjjj] = '0'
                            for kurde in range(len(state_here)):
                                new_state[all_qubits_outside_ij[kurde]] = state_here[kurde]
                            new_state = ''.join([x for x in new_state])
                            possible_states_ij.append(new_state)

                        map_cluster = anf.get_enumerated_rev_map_from_indices(
                            sorted(clu_i + clu_j))

                        matrices_cluster_i_proper, matrices_cluster_j_proper = copy.deepcopy(
                            matrices_cluster_i), copy.deepcopy(matrices_cluster_j)
                        if len(deps_i) == 0:
                            matrices_cluster_i_proper['neighbours'] = None
                        else:
                            matrices_cluster_i_proper['neighbours'] = deps_i

                        if len(deps_j) == 0:
                            matrices_cluster_j_proper['neighbours'] = None
                        else:
                            matrices_cluster_j_proper['neighbours'] = deps_j

                        mitigation_errors_ij = 0
                        for state_possible in possible_states_ij:
                            big_lambda_creator_now = GlobalNoiseMatrixCreator(
                                properly_formatted_lambdas)

                            big_matrix_now = self.create_big_lambda_modified(
                                [matrices_cluster_i_proper, matrices_cluster_j_proper],
                                [clu_i, clu_j],
                                state_possible,
                                map_cluster
                            )
                            err_now = np.linalg.norm(averaged_matrix_clusters_i_j - big_matrix_now,
                                                     ord=1)
                            # print(err_now)
                            if err_now > mitigation_errors_ij:
                                mitigation_errors_ij = err_now

                        # TODO: Does this work already?

            # check whether qubits are properly sorted
            sorted_quest = True
            if cluster_i == cluster_j:
                pass
            else:
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

                rev_map = anf.get_enumerated_rev_map_from_indices(cluster_i + cluster_j)

                qubits_in_here_dict = dict(enumerate(qubits_in_here))

                while qubits_in_here_dict != sorted_qubits_in_here:
                    for index_qubit_hehe in range(len(qubits_in_here) - 1):
                        if qubits_in_here[index_qubit_hehe] < qubits_in_here[index_qubit_hehe + 1]:
                            pass
                        elif qubits_in_here[index_qubit_hehe + 1] < qubits_in_here[
                            index_qubit_hehe]:
                            averaged_matrix = qrem_pt.permute_matrix(averaged_matrix,
                                                                     len(whole_marginal),
                                                                     [index_qubit_hehe + 1,
                                                                      index_qubit_hehe + 2])

                            anf.cool_print('Swapping qubits:', )
                            print(qubits_in_here[index_qubit_hehe],
                                  qubits_in_here[index_qubit_hehe + 1], index_qubit_hehe,
                                  index_qubit_hehe + 1)

                            qubits_in_here[index_qubit_hehe], qubits_in_here[
                                index_qubit_hehe + 1] = qubits_in_here[index_qubit_hehe + 1], \
                                                        qubits_in_here[index_qubit_hehe]

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

            if calculate_mitigation_errors:
                # print(string_marginal, mitigation_errors_ij)
                self._mitigation_errors[string_marginal] = mitigation_errors_ij

                if mitigation_errors_ij / 2 >= 0.04:
                    highly_correlated_qubits.append(
                        {'qubits': string_marginal, 'error': mitigation_errors_ij / 2})

    def get_pairs_correction_data(self,
                                  pairs_list: list,
                                  show_progress_bar: Optional[bool] = False):

        # TODO FBM: possibly change resetting
        self._noise_matrices = {}
        self._correction_matrices = {}
        self._mitigation_errors = {}
        self._correction_indices = {}

        if show_progress_bar:
            from tqdm import tqdm

            for pair_index in tqdm(range(len(pairs_list))):
                pair = pairs_list[pair_index]
                pair_key = self.get_qubits_key(pair)
                if pair_key not in self._correction_indices.keys():
                    self.compute_pairs_correction_matrices([pair], False)
        else:
            for pair in pairs_list:
                pair_key = self.get_qubits_key(pair)

                if pair_key not in self._correction_indices.keys():
                    self.compute_pairs_correction_matrices([pair], False)

        correction_data = {'correction_matrices': self._correction_matrices,
                           'noise_matrices': self._noise_matrices,
                           'correction_indices': self._correction_indices}

        return correction_data
