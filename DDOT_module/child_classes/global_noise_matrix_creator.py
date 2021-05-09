"""
Created on 29.04.2021

@author: Filip Maciejewski
@contact: filip.b.maciejewski@gmail.com
"""
import numpy as np
from typing import Optional


class GlobalNoiseMatrixCreator:
    def __init__(self,
                 noise_matrices_dictionary: dict,
                 clusters_dictionary: Optional[dict] = None,
                 neighborhoods: Optional[dict] = None,
                 ) -> None:

        self._noise_matrices_dictionary = noise_matrices_dictionary
        if clusters_dictionary is None:
            clusters_dictionary = {}
        if neighborhoods is None:
            neighborhoods = {}

        self._clusters_dictionary = clusters_dictionary
        self._neighborhoods = neighborhoods

        self._matrix_elements_dictionary = {}

    @staticmethod
    def get_qubits_key(list_of_qubits):
        return 'q' + 'q'.join([str(s) for s in list_of_qubits])

    def compute_global_noise_matrix(self,
                                    clusters_list: list,
                                    neighbors_of_clusters: Optional[list] = None,
                                    state_of_neighbors: Optional[str] = None,
                                    mapping_cluster_qubits: Optional[dict] = None):

        if neighbors_of_clusters is None:
            neighbors_of_clusters = []
            for clust in clusters_list:
                neighbors_of_clusters.append(self._neighborhoods[self.get_qubits_key(clust)])




        lambdas = [self._noise_matrices_dictionary[self.get_qubits_key(clust)] for clust in
                   clusters_list]
        number_of_qubits = sum([len(inds) for inds in clusters_list])

        if mapping_cluster_qubits is None:
            mapping_cluster_qubits = {q: q for q in range(50)}

        d = int(2 ** number_of_qubits)

        big_lambda = np.zeros((d, d))

        for input_state_integer in range(d):
            ideal_state = "{0:b}".format(input_state_integer).zfill(number_of_qubits)
            for measured_state_integer in range(d):
                measured_state = "{0:b}".format(measured_state_integer).zfill(number_of_qubits)

                element = 1
                for cluster_index in range(len(lambdas)):
                    indices_of_interest_now = clusters_list[cluster_index]

                    neighbours_now = neighbors_of_clusters[cluster_index]
                    # print(neighbours_now)

                    if neighbours_now is not None:
                        # intersection_now = anf.lists_intersection_multi(
                        #     [neighbours_now] + clusters_list)
                        # neighbours_string_ideal0 = [state_of_neighbors[x] for x in neighbours_now]
                        #
                        # # TODO: something weird here
                        # for b in intersection_now:
                        #     neighbours_string_ideal0[mapping_cluster_qubits[b]] = ideal_state[
                        #         mapping_cluster_qubits[b]]
                        #
                        #
                        if state_of_neighbors is None:
                            input_state_neighbors = ''.join([ideal_state[a] for a in neighbours_now])
                        else:
                            input_state_neighbors = state_of_neighbors

                        neighbors_string = self.get_qubits_key(neighbours_now)

                        # print(ideal_state, neighbours_now, neighbors_string, input_state_neighbors)
                        # print(lambdas[cluster_index])
                        # print(lambdas[cluster_index].keys())
                        lambda_of_interest = lambdas[cluster_index][neighbors_string][
                            input_state_neighbors]

                    else:
                        # print(lambdas[cluster_index])
                        try:
                            lambda_of_interest = lambdas[cluster_index]['averaged']
                        except(KeyError):
                            try:
                                print(cluster_index, clusters_list[cluster_index], neighbours_now,
                                      lambdas)
                                lambda_of_interest = lambdas[cluster_index]['']
                            except(KeyError):
                                # print('this:',neighbours_now,lambdas[cluster_index])
                                raise KeyError('Something wrong with averaged lambda')

                    small_string_ideal = ''.join(
                        [list(ideal_state)[mapping_cluster_qubits[b]] for b in
                         indices_of_interest_now])
                    small_string_measured = ''.join(
                        [list(measured_state)[mapping_cluster_qubits[b]] for b in
                         indices_of_interest_now])

                    element *= lambda_of_interest[
                        int(small_string_measured, 2), int(small_string_ideal, 2)]

                big_lambda[measured_state_integer, input_state_integer] = element

        return big_lambda
