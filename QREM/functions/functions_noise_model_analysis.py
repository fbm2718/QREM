"""
@authors: Filip Maciejewski, Oskar Słowik, Tomek Rybotycki
@contact: filip.b.maciejewski@gmail.com
"""

import copy
from functions import povmtools, ancillary_functions as anf


def get_neighborhood_treshold_statitsical_pairs(number_of_samples,
                                                number_of_qubits=1,
                                                probability_of_error=0.001):
    pairs_number = number_of_qubits * (number_of_qubits - 1) / 2
    eps1q = povmtools.get_statistical_error_bound(2,
                                                  number_of_samples,
                                                  probability_of_error,
                                                  pairs_number)

    return 2 * eps1q


def rename_clusters(clusters, true_indices):
    proper_clusters = {}
    for key, value in clusters.items():
        indices_enum_key = anf.get_qubit_indices_from_string(key)
        indices_enum_value = value

        key_true = ''.join(['q%s' % true_indices[qi] for qi in indices_enum_key])
        value_true = [true_indices[qi] for qi in indices_enum_value]

        proper_clusters[key_true] = value_true

    return proper_clusters


def cut_subset_sizes(clusters_neighbourhoods_dict,
                     correlations_table,
                     target_size=5):
    cutted_dict = copy.deepcopy(clusters_neighbourhoods_dict)

    for cluster, neighbours in clusters_neighbourhoods_dict.items():
        correlations_now = []
        cluster_inds = anf.get_qubit_indices_from_string(cluster)
        for ni in neighbours:
            for ci in cluster_inds:
                correlations_now.append([ni, correlations_table[ci, ni]])

        sorted_correlations = sorted(correlations_now, key=lambda x: x[1], reverse=True)

        base_size = len(cluster_inds)

        cut_neighbourhood = []

        if base_size == target_size:
            pass
        else:
            for tup in sorted_correlations:
                if (base_size + len(cut_neighbourhood)) == target_size:
                    # print()
                    break
                else:
                    ni = tup[0]
                    if ni not in cut_neighbourhood:
                        cut_neighbourhood.append(ni)

        cutted_dict[cluster] = sorted(cut_neighbourhood)

    return cutted_dict
