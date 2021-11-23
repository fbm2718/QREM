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
from functions import ancillary_functions as anf
import numpy as np
from typing import Optional, Dict, List, Union


class GlobalNoiseMatrixCreator:
    """
    This is class that, given noise matrices on clusters of qubits as function of input state of their
    neighbors, constructs global noise model on all qubits.
    """

    def __init__(self,
                 noise_matrices_dictionary: Dict[
                     str, Union[np.ndarray, Dict[str, Dict[str, np.ndarray]]]],
                 clusters_list: List[List[int]],
                 neighborhoods: Dict[str, List[int]]
                 ) -> None:

        self._noise_matrices_dictionary = noise_matrices_dictionary
        self._clusters_list = clusters_list

        for cluster_key, neighbors_list in neighborhoods.items():
            if neighbors_list is not None:
                if len(neighbors_list) == 0:
                    neighborhoods[cluster_key] = None

        self._neighborhoods = neighborhoods

        clusters_labels_list, neighbors_labels_list = [], []
        for cluster_index in range(len(self._clusters_list)):
            key_cluster = self.get_qubits_key(self._clusters_list[cluster_index])
            clusters_labels_list.append(key_cluster)
            neighbors_labels_list.append(self.get_qubits_key(self._neighborhoods[key_cluster]))

        self._clusters_labels_list = clusters_labels_list
        self._neighbors_labels_list = neighbors_labels_list

        self._matrix_elements_dictionary = {}
        self._number_of_qubits = sum([len(indices) for indices in clusters_list])
        self._global_noise_matrix = np.zeros((self._number_of_qubits, self._number_of_qubits),
                                             dtype=float)

    @staticmethod
    def get_qubits_key(list_of_qubits):
        if list_of_qubits is None:
            return None
        return 'q' + 'q'.join([str(s) for s in list_of_qubits])

    def update_labels_lists(self):
        clusters_labels_list, neighbors_labels_list = [], []
        for cluster_index in range(len(self._clusters_list)):
            key_cluster = self.get_qubits_key(self._clusters_list[cluster_index])
            clusters_labels_list.append(key_cluster)
            neighbors_labels_list.append(self.get_qubits_key(self._neighborhoods[key_cluster]))

        self._clusters_labels_list = clusters_labels_list
        self._neighbors_labels_list = neighbors_labels_list

    def compute_matrix_element(self,
                               input_state: str,
                               output_state: str):
        """
        Function that computes single global noise matrix element.

       :param input_state: bitstring denoting INPUT classical state
       :param output_state: bitstring denoting OUTPUT classical state
        """

        matrix_element = 1
        for cluster_index in range(len(self._clusters_list)):
            cluster_label_now = self._clusters_labels_list[cluster_index]
            neighbors_now = self._neighborhoods[cluster_label_now]
            neighbors_label_now = self._neighbors_labels_list[cluster_index]
            qubits_now = self._clusters_list[cluster_index]

            if neighbors_now is None:
                neighbors_input_state_now = 'averaged'
            else:
                neighbors_input_state_now = ''.join([input_state[s] for s in neighbors_now])

            cluster_input_state = ''.join([input_state[s] for s in qubits_now])
            cluster_output_state = ''.join([output_state[s] for s in qubits_now])

            if neighbors_label_now in self._noise_matrices_dictionary[cluster_label_now].keys():
                cluster_matrix = \
                    self._noise_matrices_dictionary[cluster_label_now][neighbors_label_now][
                        neighbors_input_state_now]
            else:
                if neighbors_now is None:
                    try:
                        cluster_matrix = self._noise_matrices_dictionary[cluster_label_now][
                            'averaged']
                    except(KeyError):
                        cluster_matrix = self._noise_matrices_dictionary[cluster_label_now][
                            '']
                else:
                    cluster_matrix = self._noise_matrices_dictionary[cluster_label_now][
                        neighbors_input_state_now]

            matrix_element *= cluster_matrix[int(cluster_output_state, 2),
                                             int(cluster_input_state, 2)]

        return matrix_element

    def compute_global_noise_matrix(self):
        """
            This method_name is faster than other one
        """
        number_of_qubits = self._number_of_qubits

        dimension = int(2 ** number_of_qubits)

        classical_register = anf.register_names_qubits(range(number_of_qubits))

        self._global_noise_matrix = np.zeros((dimension, dimension))

        for input_state_bitstring in classical_register:
            for output_state_bitstring in classical_register:
                self._global_noise_matrix[int(output_state_bitstring, 2),
                                          int(input_state_bitstring, 2)] = \
                    self.compute_matrix_element(input_state_bitstring, output_state_bitstring)

        return self._global_noise_matrix

    def compute_global_noise_matrix_old(self,
                                        clusters_list: Optional[List[List[int]]] = None,
                                        neighbors_of_clusters: Optional[List[List[int]]] = None):
        updated_lists = False
        if clusters_list is None:
            if self._clusters_list is None:
                raise ValueError('Please provide clusters list.')
            else:
                self.update_labels_lists()
                updated_lists = True

            clusters_list = self._clusters_list

        if neighbors_of_clusters is None:
            if self._neighborhoods is None:
                raise ValueError('Please provide neighbors list')
            else:
                if not updated_lists:
                    self.update_labels_lists()

                neighbors_of_clusters = [self._neighborhoods[self.get_qubits_key(clust)] for clust in
                                         clusters_list]

        lambdas = [self._noise_matrices_dictionary[self.get_qubits_key(clust)] for clust in
                   clusters_list]
        number_of_qubits = sum([len(inds) for inds in clusters_list])

        d = int(2 ** number_of_qubits)

        big_lambda = np.zeros((d, d))

        for input_state_integer in range(d):
            input_state_bitstring = "{0:b}".format(input_state_integer).zfill(number_of_qubits)
            for output_state_integer in range(d):
                output_state_bitstring = "{0:b}".format(output_state_integer).zfill(number_of_qubits)

                element = 1
                for cluster_index in range(len(lambdas)):
                    qubits_now = clusters_list[cluster_index]
                    neighbours_now = neighbors_of_clusters[cluster_index]

                    if neighbours_now is not None:
                        input_state_neighbors = ''.join(
                            [input_state_bitstring[a] for a in neighbours_now])
                        neighbors_string = self.get_qubits_key(neighbours_now)
                        if neighbors_string in lambdas[cluster_index].keys():
                            lambda_of_interest = lambdas[cluster_index][neighbors_string][
                                input_state_neighbors]
                        else:
                            lambda_of_interest = lambdas[cluster_index][
                                input_state_neighbors]
                    else:
                        try:
                            lambda_of_interest = lambdas[cluster_index]['averaged']
                        except(KeyError):
                            try:
                                lambda_of_interest = lambdas[cluster_index]['']
                            except(KeyError):
                                raise KeyError('Something wrong with averaged lambda')

                    small_string_ideal = ''.join(
                        [list(input_state_bitstring)[b] for b in qubits_now])
                    small_string_measured = ''.join(
                        [list(output_state_bitstring)[b] for b in qubits_now])

                    element *= lambda_of_interest[
                        int(small_string_measured, 2), int(small_string_ideal, 2)]

                big_lambda[output_state_integer, input_state_integer] = element

        return big_lambda
