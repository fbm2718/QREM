"""
Created on 29.04.2021

@author: Filip Maciejewski
@contact: filip.b.maciejewski@gmail.com
"""

import numpy as np
import copy
import QREM.ancillary_functions as anf
from typing import Optional, List, Dict, Union
from QREM.povmtools import get_enumerated_rev_map_from_indices
from QREM.DDOT_module.child_classes.ddot_marginal_analyzer_vanilla import DDTMarginalsAnalyzer
from DDOT_module.functions.functions_noise_model_heuristic import partition_algorithm_v1_cummulative

from povms_qi import povmtools


class NoiseModelGenerator(DDTMarginalsAnalyzer):
    """
        1
    """

    def __init__(self,
                 results_dictionary_ddot: Dict[str, Dict[str, int]],
                 bitstrings_right_to_left: bool,
                 number_of_qubits: int,
                 marginals_dictionary: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
                 noise_matrices_dictionary: Optional[
                     Dict[str, Union[np.ndarray, Dict[str, Dict[str, np.ndarray]]]]] = None,
                 clusters_list: Optional[List[List[int]]] = None,
                 neighborhoods: Dict[str, List[int]] = None
                 ) -> None:

        super().__init__(results_dictionary_ddot,
                         bitstrings_right_to_left,
                         marginals_dictionary,
                         noise_matrices_dictionary
                         )
        self._number_of_qubits = number_of_qubits
        self._qubit_indices = list(range(number_of_qubits))

        self._correlations_table_pairs = None

        if clusters_list is None:
            clusters_list = []

        if neighborhoods is None:
            neighborhoods = {}

        self._clusters_list = clusters_list

        self._neighborhoods = neighborhoods

    @property
    def correlations_table_pairs(self) -> np.ndarray:
        return self._correlations_table_pairs

    @correlations_table_pairs.setter
    def correlations_table_pairs(self, correlations_table_pairs: np.ndarray) -> None:
        self._correlations_table_pairs = correlations_table_pairs

    @property
    def clusters_list(self) -> List[List[int]]:
        return self._clusters_list

    @clusters_list.setter
    def clusters_list(self, clusters_list: List[List[int]]) -> None:
        for cluster in clusters_list:
            cluster_string = self.get_qubits_key(cluster)
            if cluster_string not in self._noise_matrices_dictionary.keys():
                average_noise_matrix_now = self._compute_noise_matrix_averaged(cluster)
                dictionary_now = {'averaged': average_noise_matrix_now}

                if cluster_string in self._neighborhoods.keys():
                    neighborhood_now = self._neighborhoods[cluster_string]
                    dependent_noise_matrices = self._compute_noise_matrix_dependent(cluster,
                                                                                    neighborhood_now)
                    dictionary_now = {**dictionary_now, **dependent_noise_matrices}
                    anf.cool_print('im doing this')

                self._noise_matrices_dictionary[self.get_qubits_key(cluster)] = dictionary_now

        self._clusters_list = clusters_list

    @property
    def neighborhoods(self) -> Dict[str, List[int]]:
        return self._neighborhoods

    @neighborhoods.setter
    def neighborhoods(self, neighborhoods: Dict[str, List[int]]) -> None:
        self._neighborhoods = neighborhoods
        self.clusters_list = [self.get_qubit_indices_from_string(cluster_string) for cluster_string in
                              neighborhoods.keys()]

        for cluster_string in neighborhoods.keys():
            dictionary_now = self._noise_matrices_dictionary[cluster_string]
            neighborhood_now = neighborhoods[cluster_string]
            # print(dictionary_now.keys())
            neighbors_key = self.get_qubits_key(neighborhood_now)
            if neighbors_key not in dictionary_now.keys():
                cluster = anf.get_qubit_indices_from_string(cluster_string)

                dependent_noise_matrices = self._compute_noise_matrix_dependent(cluster,
                                                                                neighborhood_now)

                self._noise_matrices_dictionary[cluster_string] = {**dictionary_now,
                                                                   **dependent_noise_matrices}

    def compute_correlations_table_pairs(self,
                                         qubit_indices: Optional[List[int]] = None,
                                         chopping_threshold: Optional[float] = 0.) -> np.ndarray:
        """From marginal noise matrices, get correlations between pairs of qubits.
           Correlations are defined as:

           c_{j -> i} = 1/2 * || \Lambda_{i}^{Y_j = '0'} - \Lambda_{i}^{Y_j = '0'}||_{l1}

           Where \Lambda_{i}^{Y_j} is an effective noise matrix on qubit "i" (averaged over all other of
           qubits except "j"), provided that input state of qubit "j" was "Y_j". Hence, c_{j -> i}
           measures how much noise on qubit "i" depends on the input state of qubit "j".

           :param qubit_indices: list of integers labeling the qubits we want to consider
                  if not provided, uses class property self._qubit_indices

           :param chopping_threshold: numerical value, for which correlations lower than
                  chopping_threshold are set to 0. If not provided, does not chop. In general, it is a
                  advisable to set such cluster_threshold that cuts off values below expected statistical
                  fluctuations.

           :return: correlations_table (ARRAY):
                    element correlations_table[i,j] = how qubit "j" AFFECTS qubit "i" [= how noise on qubit "i" depends on "j"]
           """

        add_property = False
        if qubit_indices is None:
            add_property = True
            qubit_indices = self._qubit_indices

        number_of_qubits = len(qubit_indices)
        correlations_table = np.zeros((number_of_qubits, number_of_qubits))

        if np.max(qubit_indices) > number_of_qubits:
            mapping = get_enumerated_rev_map_from_indices(qubit_indices)
        else:
            mapping = {qi: qi for qi in qubit_indices}

        for qi in qubit_indices:
            for qj in qubit_indices:
                ha, he = mapping[qi], mapping[qj]
                if qj > qi:
                    lam_i_j = self.get_noise_matrix_dependent([qi], [qj])
                    lam_j_i = self.get_noise_matrix_dependent([qj], [qi])

                    diff_i_j = lam_i_j['0'] - lam_i_j['1']
                    diff_j_i = lam_j_i['1'] - lam_j_i['0']

                    correlation_i_j = 1 / 2 * np.linalg.norm(diff_i_j, ord=1)
                    correlation_j_i = 1 / 2 * np.linalg.norm(diff_j_i, ord=1)

                    if correlation_i_j >= chopping_threshold:
                        correlations_table[ha, he] = correlation_i_j

                    if correlation_j_i >= chopping_threshold:
                        correlations_table[he, ha] = correlation_j_i

        if add_property:
            self._correlations_table_pairs = correlations_table

        return correlations_table

    def _compute_clusters_naive(self,
                                cluster_threshold: float,
                                max_size: int) -> list:
        """
            Get partition of qubits in a device into disjoint "clusters". This function uses "naive"
            method by assigning qubits to the same cluster if correlations between them are higher
            than some "threshold". It restricts size of the cluster to "maximal_size" by disregarding
            the lowest correlations (that are above threshold).
            It uses table of correlations from class property self._correlations_table_pairs

          :param cluster_threshold: correlations magnitude above which qubits are assigned
                 to the same cluster
          :param max_size: maximal allowed size of the cluster

          :return: clusters_list: list of lists, each representing a single cluster
             """
        self._clusters_list = []

        qubit_indices = self._qubit_indices
        # number_of_qubits = len(qubit_indices)

        clusters = {'q%s' % qi: [[qi, 0., 0.]] for qi in qubit_indices}
        for qi in qubit_indices:
            for qj in qubit_indices:
                if qj > qi:
                    corr_j_i, corr_i_j = self._correlations_table_pairs[qj, qi], \
                                         self._correlations_table_pairs[qi, qj]

                    # if any of the qubit affects the other strongly enough,
                    # we assign them to the same cluster
                    if corr_j_i >= cluster_threshold or corr_i_j >= cluster_threshold:
                        clusters['q%s' % qi].append([qj, corr_i_j, corr_j_i])
                        clusters['q%s' % qj].append([qi, corr_i_j, corr_j_i])

        # Merge clusters containing the same qubits
        new_lists = []
        for key, value in clusters.items():
            clusters[key] = sorted(value, key=lambda x: x[0])
            new_lists.append([vi[0] for vi in clusters[key]])

        while anf.check_if_there_are_common_elements(new_lists):
            for i in range(len(new_lists)):
                cl0 = new_lists[i]
                for j in range(len(new_lists)):
                    cl1 = new_lists[j]
                    if len(anf.lists_intersection(cl0, cl1)) != 0:
                        new_lists[i] = anf.lists_sum(cl0, cl1)

            unique_stuff = [sorted(l) for l in np.unique(new_lists)]
            new_lists = copy.deepcopy(unique_stuff)

        clusters_list = new_lists

        # Chop clusters if they exceed max size
        chopped_clusters = []
        for cluster in clusters_list:
            if len(cluster) > max_size:
                correlations_sorting = []
                for qi in cluster:
                    # as figure of merit, we will sum all correlations that are between
                    # given qubit and other guys in its cluster.
                    x = 0.0
                    for list_now in clusters['q%s' % qi]:
                        x += np.max([list_now[1], list_now[2]])

                    correlations_sorting.append([qi, x])

                correlations_sorted = sorted(correlations_sorting, key=lambda arg: arg[1],
                                             reverse=True)

                # choose only "maximal_size" qubits to belong to given cluster
                qubits_sorted = [correlations_sorted[index][0] for index in range(max_size)]
            else:
                qubits_sorted = cluster
            chopped_clusters.append(qubits_sorted)

        chopped_clusters_sorted = sorted(chopped_clusters, key=lambda y: y[0])

        self._clusters_list = chopped_clusters_sorted

        return chopped_clusters_sorted


    def _find_neighbors_of_cluster_naive(self,
                                         cluster: List[int],
                                         maximal_size: int,
                                         threshold: float
                                         ) -> List[int]:

        qubit_indices = self._qubit_indices

        potential_neighbors = []

        for qj in qubit_indices:
            affections_qj = []
            for qi in cluster:
                if qj not in cluster:
                    corr_j_i = self._correlations_table_pairs[qi, qj]
                    # if any of the qubit affects the other strongly enough,
                    # we assign them to the same cluster
                    # if cluster == [4, 5]:
                    #     print(qj, corr_j_i)

                    affections_qj.append(corr_j_i)
            if qj not in cluster:
                # print(affections_qj)
                corr_j_i = np.max(affections_qj)

                if corr_j_i >= threshold:
                    potential_neighbors.append([qj, corr_j_i])
        #
        # if cluster == [4, 5]:
        #     raise KeyError

        sorted_neighbors = sorted(potential_neighbors, key=lambda x: x[1], reverse=True)

        target_size = maximal_size - len(cluster)

        # print(sorted_neighbors)

        range_final = int(np.min([len(sorted_neighbors), target_size]))

        # if len(sorted_neighbors) <= target_size:
        return sorted([sorted_neighbors[index][0] for index in
                       range(range_final)])

        # else:
        #     cutted_neighbors = [sorted_neighbors[index][0] for index in
        #                         range(target_size)]
        #
        #     return cutted_neighbors

    def _find_neighbors_of_cluster_hollistic(self,
                                             cluster: List[int],
                                             maximal_size: int,
                                             chopping_threshold: Optional[float] = 0.) -> List[int]:
        """
        For a given cluster of qubits, find qubits which are their all_neighbors, i.e., they affect the
        noise matrix of cluster significantly. Figure of merit for correlations here is:

        c_{j -> cluster} = 1/2 || \Lambda_{cluster}^{Y_j='0'}- \Lambda_{cluster}^{Y_j='1'}||_{l1}

        where \Lambda_{cluster}^{Y_j} is the noise matrix describing noise on qubits in "cluster"
        provided that input state of qubit "j" was "Y_j".
        See also description of self._compute_clusters_naive.


        :param cluster: list of labels of qubits in a cluster
        :param maximal_size: maximal allowed size of the set "cluster+neighborhood"
        :param chopping_threshold: numerical value, for which correlations lower than
              chopping_threshold are set to 0. If not provided, it adds all_neighbors until maximal_size
              is met.



        :return: neighbors_list: list of lists, each representing a single cluster
        """

        size_cut = maximal_size - len(cluster)

        potential_neighbours = []
        for qi in self._qubit_indices:
            if qi not in cluster:
                lam_ci_j = self.get_noise_matrix_dependent(cluster,
                                                           [qi])
                diff_ci_j = lam_ci_j['0'] - lam_ci_j['1']
                correlation_ci_j = 1 / 2 * np.linalg.norm(diff_ci_j, ord=1)
                potential_neighbours.append([qi, correlation_ci_j])

        sorted_neighbours = sorted(potential_neighbours, key=lambda x: x[1], reverse=True)

        neighbors_list = sorted(
            [sorted_neighbours[i][0] for i in range(int(np.min([size_cut, len(sorted_neighbours)]))) if
             chopping_threshold < sorted_neighbours[i][1]])

        cluster_key = self.get_qubits_key(cluster)

        self._neighborhoods[cluster_key] = neighbors_list

        return neighbors_list

    def _find_all_neighborhoods_holistic(self,
                                         maximal_size,
                                         chopping_threshold: float,
                                         show_progress_bar: Optional[bool] = False) \
            -> Dict[str, List[int]]:
        self._neighborhoods = {}
        if show_progress_bar:
            from tqdm import tqdm
            clusters_list = self._clusters_list

            for index_cluster in tqdm(range(len(clusters_list))):
                cluster = clusters_list[index_cluster]
                self._neighborhoods[
                    self.get_qubits_key(cluster)] = self._find_neighbors_of_cluster_hollistic(
                    cluster,
                    maximal_size,
                    chopping_threshold)

        else:
            for cluster in self._clusters_list:
                self._neighborhoods[
                    self.get_qubits_key(cluster)] = self._find_neighbors_of_cluster_hollistic(
                    cluster,
                    maximal_size,
                    chopping_threshold)
        return self._neighborhoods

    def _find_all_neighborhoods_pairwise(self,
                                         maximal_size: int,
                                         threshold: float
                                         ):
        self._neighborhoods = {}

        if self._correlations_table_pairs is None:
            self.compute_correlations_table_pairs()

        for cluster in self._clusters_list:
            self._neighborhoods[self.get_qubits_key(cluster)] = self._find_neighbors_of_cluster_naive(
                cluster, maximal_size=maximal_size, threshold=threshold)

        return self._neighborhoods


    def compute_clusters(self,
                         maximal_size: float,
                         method: Optional[str] = 'holistic_v1',
                         method_kwargs: Optional[dict] = None) -> list:
        """
        Get partition of qubits in a device into disjoint "clusters".
        This function uses various heuristic methods, specified via string "version".
        It uses table of correlations from class property self._correlations_table_pairs

        :param maximal_size: maximal allowed size of the cluster
        :param method: string specifying type of heuristic
        Possible values:
            'pairwise' - heuristic that uses Algorithm 3 from Ref.[]
            'holistic_v1' - heuristic that uses function partition_algorithm_v1_cummulative

        :param method_kwargs: potential arguments that will be passed to clustering function.
                           For possible parameters see descriptions of particular functions.

        :return: clusters_list: list of lists, each representing a single cluster
        """
        self._clusters_list = []

        if method == 'pairwise':
            if method_kwargs is None:
                default_kwargs = {'cluster_threshold': 0.02,
                                  'maximal_size': maximal_size}

                method_kwargs = default_kwargs

            clusters_list = self._compute_clusters_naive(**method_kwargs)


        elif method == 'holistic_v1':
            if method_kwargs is None:
                alpha = 1
                algorithm_runs = 1000
                default_kwargs = {'alpha': alpha,
                                  'N_alg': algorithm_runs,
                                  'printing': False,
                                  'drawing': False}

                method_kwargs = default_kwargs

            method_kwargs['C_maxsize'] = maximal_size
            clusters_list, score = partition_algorithm_v1_cummulative(self._correlations_table_pairs,
                                                                      **method_kwargs)

            anf.cool_print('Current partitioning got score:', score)
        else:
            raise ValueError('No heuristic with that name: ' + method)

        self._clusters_list = clusters_list

        return clusters_list

    def find_all_neighborhoods(self,
                               maximal_size: int,
                               method: Optional[str] = 'holistic',
                               method_kwargs: Optional[dict] = None):

        if method == 'pairwise':
            if method_kwargs is None:
                default_kwargs = {'threshold': 0.01}
                method_kwargs = default_kwargs

            method_kwargs['maximal_size'] = maximal_size
            neighborhoods = self._find_all_neighborhoods_pairwise(**method_kwargs)

        elif method == 'holistic':
            if method_kwargs is None:
                default_kwargs = {'chopping_threshold': 0.0,
                                  'show_progress_bar': True}
                method_kwargs = default_kwargs
            method_kwargs['maximal_size'] = maximal_size
            neighborhoods = self._find_all_neighborhoods_holistic(**method_kwargs)

        else:
            raise ValueError('Wrong method name')

        return neighborhoods

    def print_properties(self):
        # TODO FBM, OS: add this

        return None

    def draw_noise_model(self):
        # TODO FBM, OS: add this

        return None
