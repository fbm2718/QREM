"""
Created on 03.05.2021

@author: Filip Maciejewski
@contact: filip.b.maciejewski@gmail.com
"""

from QREM import ancillary_functions as anf
import copy
import numpy as np
from QREM.DDOT_module.functions import depreciated_functions as df
import QREM

"""
functions for data analysis
"""

def get_noise_matrix_from_counts_dict(results_dictionary):
    """Return noise matrix from counts dictionary.
    Assuming that the results are given only for qubits of interest.
    :param results_dictionary (dictionary): results dictionary for which KEY is the bitstring denoting INPUT CLASSICAL STATE, while VALUE is the probability vector of results

    :return: noise_matrix (array): the array representing noise on qubits on which the experiments were performed
    """

    number_of_qubits = len(list(results_dictionary.keys())[0])
    noise_matrix = np.zeros((2 ** number_of_qubits, 2 ** number_of_qubits))
    for input_state, probability_vector in results_dictionary.items():
        noise_matrix[:, int(input_state, 2)] = probability_vector[:, 0]
    return noise_matrix


def get_marginals_from_counts_dict(counts,
                                   marginals_dictionary,
                                   reverse_counts=False,
                                   normalize=False,
                                   qubits_mapping=None):
    """Return dictionary of marginal probability distributions from counts dictionary
    :param counts (dictionary): results dictionary for which KEY is the bitstring denoting result of measurement in computational basis, while VALUE is the number of occurances of that result
    :param marginals_dictionary (dictionary):  the dictionary which will be filled with the counts from "counts" dictionary.
    Each KEY is the string of the form "q1q5q13" specyfing what qubits we are interested in, each VALUE is a vector of the size 2**(number of qubits in marginal) which might be either filled with 0s or filled with previous values
    :param (optional) reverse_counts (Boolean): Specify whether measurement result bitsstring should be reversed before adding to marginal (this is the case for qiskit where bits are counted from right)
    :param (optional) normalize (Boolean): specify whether marginal distributions should be normalized to 1
    :param (optional) qubits_mapping (dict): optional dictionary with qubits labels mapping


    :return: marginals_dictionary (dictionary): filled marginals dictionary of the same structure as parameter "marginals_dictionary" (but now filled with values from "counts")
    """

    for qubits_string in marginals_dictionary.keys():
        qubits_indices = anf.get_qubit_indices_from_string(qubits_string)
        if qubits_mapping is not None:
            bits_of_interest = [qubits_mapping(qubits_indices[i]) for i in range(qubits_indices)]
        else:
            bits_of_interest = qubits_indices

        for count, ticks in counts.items():
            if reverse_counts:
                count = count[::-1]

            marginal_key_now = ''.join([count[b] for b in bits_of_interest])
            marginals_dictionary[qubits_string][int(marginal_key_now, 2)] += ticks

    if normalize:
        for qubits_string in marginals_dictionary.keys():
            marginals_dictionary[qubits_string] *= 1 / sum(marginals_dictionary[qubits_string])

    return marginals_dictionary


def get_subsets_marginals_from_counts(results_dictionary,
                                      subsets,
                                      reverse_counts,
                                      qubits_mapping=None):
    """Return dictionary of marginal probability distributions from results_dictionary
    :param results_dictionary (dictionary): results dictionary for which KEY is the bitstring denoting INPUT CLASSICAL STATE, while VALUE is the counts dictionary with results of the experiments
    :param subsets (list of lists of ints): list of lists. Each list contains labels of subset of qubits for which marginal distributions are to be calculated.
   es
    :param reverse_counts (Boolean): Specify whether measurement result bitsstring should be reversed before adding to marginal (this is the case for qiskit where bits are counted from right)
    :param (optional) qubits_mapping (dict): optional dictionary with qubits labels mapping


    :return: marginals_dictionary (dictionary): the dictionary for which each KEY is the same as for "results_dictionary" and each VALUE is the dictionary of marginals as returned by function "get_marginals_from_counts_dict"

    """

    marginals_dictionaries_template = {
        'q' + 'q'.join([str(s) for s in sub]): np.zeros((2 ** (int(len(sub))), 1)) for sub in subsets}
    marginal_dictionaries = {}

    for what_we_put, counts in results_dictionary.items():
        # print(what_we_put)
        marginals_dict_now = get_marginals_from_counts_dict(counts,
                                                            copy.deepcopy(
                                                                marginals_dictionaries_template),
                                                            reverse_counts=reverse_counts,
                                                            normalize=False,
                                                            qubits_mapping=qubits_mapping)

        marginal_dictionaries[what_we_put] = marginals_dict_now

    return marginal_dictionaries


def get_subset_noise_matrices_from_marginals(marginal_dictionaries,
                                             subsets,
                                             max_subset_length=7):
    """Return dictionary of marginal probability distributions from results_dictionary
    :param marginal_dictionaries (dictionary): dictionary (as returned by function "get_subsets_marginals_from_counts") for which KEY is the bitstring denoting INPUT CLASSICAL STATE, while VALUE is the dictionary with MARGINALS (as returned by function "get_marginals_from_counts_dict")
    :param subsets (list of lists of ints): list of lists. Each list contains labels of subset of qubits for which marginal distributions are to be calculated.

    :param (optional) max_subset_length (int): the length of the biggest subset, default is 7


    :return: noise_matrices (dictionary): dictionary where each KEY is the string denoting qubits subset, and VALUE is the noise matrix on acting on that subset.
    """

    marginals_dictionaries_template = {
        'q' + 'q'.join([str(s) for s in sub]): np.zeros((2 ** (int(len(sub))), 1)) for sub in subsets}

    from QREM.DDOT_module.functions import functions_data_analysis as fda

    mini_dicts_template = {i + 1: fda.get_mini_dict(i + 1) for i in range(max_subset_length)}

    marginal_dictionaries_subsets = {}
    for key_marginal in marginals_dictionaries_template.keys():
        qubits_now = anf.get_qubit_indices_from_string(key_marginal)
        mini_dict_now = copy.deepcopy(mini_dicts_template[len(qubits_now)])

        for what_we_put, dictionary_marginal in marginal_dictionaries.items():
            input_marginal = ''.join([what_we_put[x] for x in qubits_now])
            mini_dict_now[input_marginal] += dictionary_marginal[key_marginal]

        for key_small in mini_dict_now.keys():
            mini_dict_now[key_small] *= 1 / np.sum(mini_dict_now[key_small])

        marginal_dictionaries_subsets[key_marginal] = mini_dict_now

    noise_matrices = {}

    for marginal_key, marginal_dict in marginal_dictionaries_subsets.items():
        noise_matrices[marginal_key] = get_noise_matrix_from_counts_dict(marginal_dict)

    return noise_matrices


def get_small_noise_matrices_depending_on_neighbours_states(big_noise_matrix,
                                                            qubits_of_interest,
                                                            neighbours_of_interest):
    # print(qubits_of_interest,neighbours_of_interest)

    number_of_neighbours = len(neighbours_of_interest)
    number_of_qubits = len(qubits_of_interest)
    number_of_qubits_total = number_of_qubits + number_of_neighbours

    d_small_neighbours = int(2 ** number_of_neighbours)
    d_small_qubit = int(2 ** number_of_qubits)

    enumerating_qubits = anf.get_enumerated_rev_map_from_indices(qubits_of_interest)
    enumerating_neighbours = anf.get_enumerated_rev_map_from_indices(neighbours_of_interest)

    dictionary_neighbours = {
        anf.binary_integer_format(i, number_of_neighbours): np.zeros((d_small_qubit, d_small_qubit),
                                                                     dtype=float) for i in
        range(d_small_neighbours)}
    register_qubits = [anf.binary_integer_format(i, number_of_qubits) for i in range(d_small_qubit)]

    # print(dictionary_neighbours.keys())

    for state_of_neighbours_input in dictionary_neighbours.keys():
        # print(state_of_neighbours_input)
        for state_of_qubits_input in register_qubits:
            big_state_input = ''.join([state_of_qubits_input[
                                           enumerating_qubits[k]] if k in qubits_of_interest else
                                       state_of_neighbours_input[enumerating_neighbours[k]] for k in
                                       range(number_of_qubits_total)])

            # print(state_of_qubits_input,state_of_neighbours_input,big_state_input)
            # print(big_state_input)
            for state_of_qubits_output in register_qubits:
                for state_of_neighbours_output in dictionary_neighbours.keys():
                    big_state_output = ''.join([state_of_qubits_output[enumerating_qubits[
                        t]] if t in qubits_of_interest else state_of_neighbours_output[
                        enumerating_neighbours[t]] for t in range(number_of_qubits_total)])
                    # print(big_state_output)
                    # print(state_of_qubits_output,big_state_output)
                    dictionary_neighbours[state_of_neighbours_input][
                        int(state_of_qubits_output, 2), int(state_of_qubits_input, 2)] += \
                    big_noise_matrix[int(big_state_output, 2), int(big_state_input, 2)]

    # raise KeyError
    for key in dictionary_neighbours.keys():
        mat_copy = copy.deepcopy(dictionary_neighbours[key])
        for i in range(d_small_qubit):
            mat_copy[:, i] *= 1 / sum(mat_copy[:, i])

        # print(sum(dictionary_neighbours[key][:,0]))
        # print(dictionary_neighbours[key])
        # raise KeyError
        dictionary_neighbours[key] = mat_copy

    return dictionary_neighbours




"""
functions for noise model analysis
"""



def get_averaged_noise_matrix_from_big_noise_matrix(big_lambda,
                                                    qubits_of_interest):
    """From high-dimensional noise matrix return lower-dimensional effective noise matrix acting on marginal averaged over all other qubits than "qubits_of_interest"
    :param big_lambda (array): array describing stochastic (classical) noise on the whole space
    :param qubits_of_interest (list of ints): list of integers labeling the qubits in marginal we are interested in

    :return: small_lambda (array): effective noise matrix acting on qubits from "qubits_of_interest" (averaged over all other qubits)
    """

    # What is the number of qubits in the whole space
    big_N = int(np.log2(big_lambda.shape[0]))

    # What is the number of qubits in marginal of interest
    small_N = len(qubits_of_interest)

    # Normalization when averaging over states of neighbours (each with the same probability)
    normalization = 2 ** (big_N - small_N)

    # classical register on all qubits
    classical_register_big = ["{0:b}".format(i).zfill(big_N) for i in range(2 ** big_N)]

    # create dictionary of the marginal states of qubits of interest for the whole register
    # (this function is storing data which could also be calculated in situ in the loops later)
    indices_small = {}
    for s in classical_register_big:
        small_string = ''.join([list(s)[b] for b in qubits_of_interest])
        indices_small[s] = small_string

    # prepare the noise matrix on "qubits_of_interest"
    small_lambda = np.zeros((2 ** small_N, 2 ** small_N))

    # go through all classical states
    for i in range(2 ** big_N):
        for j in range(2 ** big_N):
            lambda_element = big_lambda[i, j]

            # input state of all qubits in binary format
            input_state = "{0:b}".format(j).zfill(big_N)
            # measured state of all qubits in binary format
            measured_state = "{0:b}".format(i).zfill(big_N)

            # input state of marginal qubits in binary format
            input_state_small = indices_small[input_state]
            # measured state of all qubits in binary format
            measured_state_small = indices_small[measured_state]

            # element of small lambda labeled by (measured state | input state)
            small_lambda[int(measured_state_small, 2), int(input_state_small, 2)] += lambda_element

    # normalize matrix and return
    return small_lambda / normalization


def get_small_noise_matrices_from_big_noise_matrix(big_lambda,
                                                   qubits_of_interest,
                                                   neighbors_of_interest,
                                                   mapping=None):
    """From high-dimensional noise matrix return lower-dimensional effective noise matrices acting on "qubits_of_interest" conditioned on input states of "all_neighbors of interest"
    :param big_lambda (array): array describing stochastic (classical) noise on the whole space
    :param qubits_of_interest (list of ints): list of integers labeling the qubits in marginal we are interested in
    :param neighbors_of_interest (list of ints): list of integers labeling the qubits that affet noise matrix on "qubits_of_interest"

    :param (optional) mapping (dictionary): optional dictionary mapping qubits labels


    :return: lambdas_dict (dictionary):
             dictionary with two KEYS:
             lambdas_dict['lambdas'] = DICTIONARY, where KEY is the input state of "neighbors_of_interest", and VALUE is the noise matrix acting on "qubits_of_interested" for this fixed input state of all_neighbors
             lambdas_dict['all_neighbors'] = list of all_neighbors (this either copies param "neighbors_of_interest" or returns all_neighbors mapped according to "mapping" param)

    """

    # If there are no all_neighbors, then this corresponds to averaging over all qubits except "qubits_of_interest"
    if len(neighbors_of_interest) == 0 or neighbors_of_interest is None:
        small_lambdas = {'neighbours': None,
                         'lambdas': get_averaged_noise_matrix_from_big_noise_matrix(big_lambda, qubits_of_interest)}
        return small_lambdas

    # check if there is no collision between qubits of interest and all_neighbors of interest (if there is, then the model won't be consistent)
    if len(anf.lists_intersection(qubits_of_interest, neighbors_of_interest)) != 0:
        print(qubits_of_interest, neighbors_of_interest)
        raise ValueError('Wrong indices')

    # What is the number of qubits in the whole space
    big_N = int(np.log2(big_lambda.shape[0]))

    # What is the number of qubits in marginal of interest
    small_N = len(qubits_of_interest)

    # What is the number of all_neighbors
    neighbours_N = len(neighbors_of_interest)

    # Normalization when averaging over states of non-neighbours (each with the same probability)
    normalization = 2 ** (big_N - neighbours_N - small_N)

    # classical register on all qubits
    classical_register_big = ["{0:b}".format(i).zfill(big_N) for i in range(2 ** big_N)]

    # classical register on neighbours
    classical_register_neighbours = ["{0:b}".format(i).zfill(neighbours_N) for i in range(2 ** neighbours_N)]

    # create dictionary of the marginal states of qubits of interest and all_neighbors for the whole register
    # (this function is storing data which could also be calculated in situ in the loops later)
    indices_small = {}
    for s in classical_register_big:
        small_string = ''.join([list(s)[b] for b in qubits_of_interest])
        neighbours_string = ''.join([list(s)[b] for b in neighbors_of_interest])
        # first place in list is label for state of "qubits_of_interest" and second for "neighbors_of_interest
        indices_small[s] = [small_string, neighbours_string]

    # initiate dictionary for which KEY is input state of all_neighbors and VALUE will the the corresponding noise matrix on "qubits_of_interest"
    small_lambdas = {s: np.zeros((2 ** small_N, 2 ** small_N)) for s in classical_register_neighbours}

    # go through all classical states
    for i in range(2 ** big_N):
        for j in range(2 ** big_N):
            lambda_element = big_lambda[i, j]

            # input state of all qubits in binary format
            input_state = "{0:b}".format(j).zfill(big_N)
            # measured state of all qubits in binary format
            measured_state = "{0:b}".format(i).zfill(big_N)

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

    # prepare dictionary with first key being the obtained noise matrices
    lambdas_dict = {'lambdas': small_lambdas,
                    }

    # add all_neighbors list to dictionary (either mapped or not)
    if mapping is None:
        lambdas_dict['all_neighbors'] = list(neighbors_of_interest)
    else:
        lambdas_dict['all_neighbors'] = [mapping[ni] for ni in neighbors_of_interest]

    return lambdas_dict


def calculate_correlations_pairs(marginal_noise_matrices,
                                 qubit_indices):
    """From marginal noise matrices, get correlations between pairs of qubits.
       :param marginal_noise_matrices (dictionary): array describing stochastic (classical) noise on the whole space
       :param qubit_indices (list of ints): list of integers labeling the qubits we want to consider


       :return: correlations_table (ARRAY):
                element correlations_table[i,j] = how qubit "j" AFFECTS qubit "i" [= how noise on qubit "i" depends on "j"]
       """

    number_of_qubits = len(qubit_indices)
    correlations_table = np.zeros((number_of_qubits, number_of_qubits))

    if np.max(qubit_indices) > number_of_qubits:
        mapping = QREM.povmtools.get_enumerated_rev_map_from_indices(qubit_indices)
    else:
        mapping = {qi: qi for qi in qubit_indices}

    for qi in qubit_indices:
        for qj in qubit_indices:
            ha, he = mapping[qi], mapping[qj]
            if qj > qi:
                big_lambda = marginal_noise_matrices['q%sq%s' % (qi, qj)]
                lam_i_j = get_small_noise_matrices_from_big_noise_matrix(big_lambda, [0], [1])['lambdas']
                lam_j_i = get_small_noise_matrices_from_big_noise_matrix(big_lambda, [1], [0])['lambdas']

                diff_i_j = lam_i_j['0'] - lam_i_j['1']
                diff_j_i = lam_j_i['1'] - lam_j_i['0']

                correlation_i_j = 1 / 2 * np.linalg.norm(diff_i_j, ord=1)
                correlation_j_i = 1 / 2 * np.linalg.norm(diff_j_i, ord=1)
                correlations_table[ha, he] = correlation_i_j
                correlations_table[he, ha] = correlation_j_i

    return correlations_table


def get_average_correlations(small_noise_matrices):
    correlation_i_j, correlation_j_i = 0, 0

    neighbours_states = list(small_noise_matrices.keys())
    norm = len(neighbours_states)

    # print(neighbours_states)

    for key in neighbours_states:
        lam_now = small_noise_matrices[key]

        lam_i_j = get_small_noise_matrices_from_big_noise_matrix(lam_now, [0], [1])['lambdas']
        lam_j_i = get_small_noise_matrices_from_big_noise_matrix(lam_now, [1], [0])['lambdas']
        diff_i_j = lam_i_j['0'] - lam_i_j['1']
        diff_j_i = lam_j_i['1'] - lam_j_i['0']

        # print(1/2*np.linalg.norm(diff_i_j))

        correlation_i_j += 1 / 2 * np.linalg.norm(diff_i_j, ord=1)
        correlation_j_i += 1 / 2 * np.linalg.norm(diff_j_i, ord=1)

    correlation_i_j *= 1 / norm
    correlation_j_i *= 1 / norm

    return correlation_i_j, correlation_j_i





def get_initial_clusters(qubit_indices,
                         correlations_table,
                         treshold):
    """From qubit indices and correlations table get initial proxy for structure of correlations
        :param qubit_indices (list of ints): list of integers labeling the qubits we want to consider
        :param correlations_table (array): array describing correlations between qubits expressed by a difference of
                                          effective noise matrix on qubit "i" depending on state of qubit "j"
             correlations_table[i,j] = how qubit "j" AFFECTS qubit "i" [= how noise on qubit "i" depends on "j"]
        :param treshold: when to consider qubits to belong to the same cluster

        :return: clusters_list: list of lists, each representing a single cluster
       """

    number_of_qubits = len(qubit_indices)

    #If indices of qubits are incompatible with size of the array, we assume they should be numbered in ascending order
    if np.max(qubit_indices) > number_of_qubits:
        mapping = QREM.povmtools.get_enumerated_rev_map_from_indices(qubit_indices)
    else:
        mapping = {qi: qi for qi in qubit_indices}


    clusters = {'q%s' % qi: [qi] for qi in qubit_indices}
    for qi in qubit_indices:
        for qj in qubit_indices:
            ha, he = mapping[qi], mapping[qj]
            if qj > qi:
                corr_j_i, corr_i_j = correlations_table[he, ha], correlations_table[ha, he]
                #if any of the qubit affects the other strong enough, we assign them to the same cluster
                if corr_j_i >= treshold or corr_i_j >= treshold:
                    # print(qi, qj, corr_j_i, corr_i_j)
                    clusters['q%s' % qi].append(qj)
                    clusters['q%s' % qj].append(qi)

    new_lists = []
    for key, value in clusters.items():
        clusters[key] = sorted(value)
        # print(clusters[key])
        new_lists.append(value)
    # print('old',new_lists)
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
    return clusters_list



def get_neighbours_of_cluster(dictionary,
                        cluster,
                        number_of_qubits,
                        maximal_size,
                        chop_treshold = 0.02,
                        reverse_counts=False,
                        qubits_mapping=None):
    import time
    converted_dict = dictionary
    chosen_cluster = cluster

    size_cut = maximal_size - len(chosen_cluster)

    potential_neighbours = []
    for qi in range(0,number_of_qubits):
        if qi not in chosen_cluster:
            subset_marginals_to_process = get_subsets_marginals_from_counts(converted_dict,
                                                                       [chosen_cluster+[qi]],
                                                                       reverse_counts=reverse_counts,
                                                                       qubits_mapping = qubits_mapping)
            noise_matrices_to_process = get_subset_noise_matrices_from_marginals(subset_marginals_to_process,
                                                                                            [chosen_cluster+[qi]])


            big_matrix = noise_matrices_to_process['q'+'q'.join([str(qsmall) for qsmall in chosen_cluster+[qi]])]

            t0= time.time()
            lam_ci_j = get_small_noise_matrices_from_big_noise_matrix(big_matrix, range(len(chosen_cluster)), [len(chosen_cluster)])['lambdas']

            t1=time.time()
            lam_ci_j_v2 = get_small_noise_matrices_depending_on_neighbours_states(big_matrix,range(len(chosen_cluster)), [len(chosen_cluster)])
            t2=time.time()

            print('First method:',t1-t0,'Second method:',t2-t1)



            diff_ci_j = lam_ci_j['0'] - lam_ci_j['1']
            diff_ci_j_v2 = lam_ci_j_v2['0'] - lam_ci_j_v2['1']

            correlation_ci_j = 1 / 2 * np.linalg.norm(diff_ci_j, ord=1)
            correlation_ci_j_v2 = 1 / 2 * np.linalg.norm(diff_ci_j_v2, ord=1)

            if abs(correlation_ci_j-correlation_ci_j_v2)>10**(-5):
                raise ValueError("Disagreement between methods")
            # print(correlation_ci_j,correlation_ci_j_v2)

            potential_neighbours.append([qi,correlation_ci_j])


    sorted_neighbours = sorted(potential_neighbours,key = lambda x: x[1],reverse=True)
    best_neighbours = sorted([sorted_neighbours[i][0] for i in range(size_cut) if chop_treshold<sorted_neighbours[i][1] ])

    anf.cool_print('\nCluster:',chosen_cluster)
    anf.cool_print('Best all_neighbors:',best_neighbours)

    return best_neighbours






def get_clusters_and_neighbors_from_pairs(qubit_indices,
                                          correlations_2q,
                                          neighborhood_treshold=0.01,
                                          cluster_treshold=0.04):
    #DEPRECIATED

    clusters = {'q%s' % i: [] for i in qubit_indices}

    for qi in qubit_indices:
        for qj in qubit_indices:
            cij, cji = correlations_2q[qi, qj], correlations_2q[qj, qi]
            if cij >= cluster_treshold or cji >= cluster_treshold:
                clusters['q%s' % qi].append(qj)

    clusters_list = []
    for qi in qubit_indices:
        clusters_list.append(sorted([qi] + clusters['q%s' % qi]))

    new_lists = copy.deepcopy(clusters_list)

    while anf.check_if_there_are_common_elements(new_lists):
        for i in range(len(new_lists)):
            cl0 = new_lists[i]
            for j in range(len(new_lists)):
                cl1 = new_lists[j]
                if len(anf.lists_intersection(cl0, cl1)) != 0:
                    new_lists[i] = anf.lists_sum(cl0, cl1)

        unique_stuff = [sorted(l) for l in np.unique(new_lists)]
        new_lists = copy.deepcopy(unique_stuff)

    proper_clusters = new_lists
    neighborhoods_clusters, neighborhoods_qubits = {}, {'q%s' % qi: [] for qi in qubit_indices}

    for cluster in proper_clusters:
        string_cluster = ''.join(['q%s' % qi for qi in cluster])
        neighborhoods_clusters[string_cluster] = []
        for qi in cluster:
            for qj in qubit_indices:
                if qj not in cluster:
                    if correlations_2q[qi, qj] >= neighborhood_treshold:
                        neighborhoods_clusters[string_cluster].append(qj)
                        neighborhoods_qubits['q%s' % qi].append(qj)

    for key, value in neighborhoods_clusters.items():
        neighborhoods_clusters[key] = list(np.unique(value))

    clusters_dict = {}
    for cluster in proper_clusters:
        for qi in cluster:
            clusters_dict['q%s' % qi] = sorted(anf.lists_difference(cluster, [qi]))

    return clusters_dict, neighborhoods_clusters, proper_clusters, neighborhoods_qubits



"""
functions for error mitigation
"""


def get_pairs_correction_matrices(counts_dict,
                                  qubit_indices,
                                  clusters,
                                  neighborhoods_clusters,
                                  backend_name,
                                  calculate_mitigation_errors = True,
                                  mapping_not_reversed = None):
    """Get two-qubit noise matrices for all pairs of qubits in a device
        :param counts_dict (ditionary): dictionary of DDOT results, KEY is INPUT classical state, and VALUE is dictionary
                                        of counts (results of experiment)
        :param qubit_indices (list of ints): list of integers labeling the qubits we want to consider
        :param clusters: when to consider qubits to belong to the same cluster
        :param neighborhoods_clusters: when to consider qubits to belong to the same cluster
        :param neighborhoods_clusters: when to consider qubits to belong to the same cluster

        :return: clusters_list: list of lists, each representing a single cluster
   """


    #Get correction matrices for all pairs of qubits
    number_of_qubits = len(qubit_indices)

    correction_indices = {}

    noise_matrices, correction_matrices, mitigation_errors = {}, {}, {}

    highly_correlated_qubits = []


    for i in tqdm(range(0,number_of_qubits)):
        for j in range(i+1, number_of_qubits):
            string_pair = 'q%sq%s' % (i, j)

            cluster_i = sorted([i] + clusters['q%s' % i])
            cluster_j = sorted([j] + clusters['q%s' % j])

            # print(cluster_i,cluster_j)

            string_cluster_i = ''.join(['q%s' % qi for qi in cluster_i])
            string_cluster_j = ''.join(['q%s' % qi for qi in cluster_j])

            neighborhoods_cluster_i = neighborhoods_clusters[string_cluster_i]
            neighborhoods_cluster_j = neighborhoods_clusters[string_cluster_j]

            # print(string_cluster_i,string_cluster_j)

            if len(anf.lists_intersection(cluster_i, cluster_j)) != 0:
                #Check if clusters overlap. If yes, we treat them as single, big cluster
                #and construct cluster-neighborhood noise model

                dependencies_clusters_i_j = sorted(anf.lists_sum(neighborhoods_cluster_i, neighborhoods_cluster_j))
                matrices_clusters = get_subset_matrices(counts_dict,
                                                        anf.lists_sum(cluster_i, cluster_j),
                                                        dependencies_clusters_i_j,
                                                        backend_name,
                                                        mapping=mapping_not_reversed)
                # matrices_clusters_old = get_subset_matrices_old(counts_dict,
                #                                                 anf.lists_sum(cluster_i, cluster_j),
                #                                                 dependencies_clusters_i_j,)
                #

                # print(matrices_clusters_old)
                # enumerated_qubits = dict(enumerate(sorted(anf.lists_sum_multi([cluster_i,cluster_j,dependencies_clusters_i_j]))))

                averaged_matrix_clusters_i_j = sum([lam for lam in matrices_clusters.values()]) / 2 ** (
                    len(dependencies_clusters_i_j))

                if calculate_mitigation_errors:
                    mitigation_errors_ij = 0
                    correction_norm_ij = np.linalg.norm(np.linalg.inv(averaged_matrix_clusters_i_j),ord=1)
                    # anf.cool_print('calculating','')
                    for lam in matrices_clusters.values():
                        err_now = np.linalg.norm(averaged_matrix_clusters_i_j-lam,ord=1)
                        # print(err_now)
                        if err_now>mitigation_errors_ij:
                            mitigation_errors_ij = err_now
                    # anf.cool_print('ok', '')
                    mitigation_errors_ij*=correction_norm_ij


            else:
                #Check if clusters overlap. If not, we treat them as separate clusters.
                dependencies_cluster_i = sorted(neighborhoods_clusters[string_cluster_i])
                dependencies_cluster_j = sorted(neighborhoods_clusters[string_cluster_j])

                # print(qubitt)
                matrices_cluster_i = get_subset_matrices(counts_dict,
                                                         cluster_i,
                                                         neighborhoods_cluster_i,
                                                         backend_name,
                                                         mapping=mapping_not_reversed)
                matrices_cluster_j = get_subset_matrices(counts_dict,
                                                         cluster_j,
                                                         neighborhoods_cluster_j,
                                                         backend_name,
                                                         mapping=mapping_not_reversed)

                intersection_i, intersection_j = anf.lists_intersection(dependencies_cluster_i,cluster_j),\
                                                 anf.lists_intersection(dependencies_cluster_j,cluster_i)


                # matrices_cluster_j_old = get_subset_matrices_old(counts_dict,cluster_j,neighborhoods_cluster_j)
                # #
                # # print(matrices_cluster_j)
                # print(matrices_cluster_j_old)

                # raise KeyboardInterrupt

                if len(intersection_i)==0 and len(intersection_j)==0:

                    # print(cluster_i,neighborhoods_cluster_i)
                    # print(matrices_cluster_i)

                    #Check if clusters contain each others all_neighbors. If not, the noise matrix is simply a tensor product of clusters.
                    averaged_matrix_cluster_i = sum([lam_i for lam_i in matrices_cluster_i.values()]) / 2 ** (
                        len(dependencies_cluster_i))

                    averaged_matrix_cluster_j = sum([lam_j for lam_j in matrices_cluster_j.values()]) / 2 ** (
                        len(dependencies_cluster_j))

                    averaged_matrix_clusters_i_j = np.kron(averaged_matrix_cluster_i, averaged_matrix_cluster_j)


                    if calculate_mitigation_errors:

                        # anf.cool_print('calculating', '')
                        mitigation_errors_i, mitigation_errors_j = 0., 0.

                        correction_norm_i, correction_norm_j = np.linalg.norm(np.linalg.inv(averaged_matrix_cluster_i),ord=1), \
                                                               np.linalg.norm(np.linalg.inv(averaged_matrix_cluster_j),ord=1)


                        for lam in matrices_cluster_i.values():
                            err_now = np.linalg.norm(averaged_matrix_cluster_i - lam, ord=1)
                            # print(err_now)
                            if err_now > mitigation_errors_i:
                                mitigation_errors_i = err_now

                        for lam in matrices_cluster_j.values():
                            err_now = np.linalg.norm(averaged_matrix_cluster_j - lam, ord=1)
                            # print(err_now)
                            if err_now > mitigation_errors_j:
                                mitigation_errors_j = err_now

                        if mitigation_errors_i>0 and mitigation_errors_j>0:
                            mitigation_errors_ij = mitigation_errors_i*mitigation_errors_j*correction_norm_i*correction_norm_j
                        elif mitigation_errors_i>0 and mitigation_errors_j==0:
                            mitigation_errors_ij = mitigation_errors_i*correction_norm_i
                        elif mitigation_errors_i==0 and mitigation_errors_j>0:
                            mitigation_errors_ij = mitigation_errors_j*correction_norm_j
                        else:
                            mitigation_errors_ij = 0


                        # print(mitigation_errors_ij)
                        # print(mitigation_errors_i,mitigation_errors_j)
                        # print(correction_norm_i,correction_norm_j)
                        # anf.cool_print('Ok','')
                        # raise KeyboardInterrupt



                else:
                    #Check if clusters are each others all_neighbors. If yes, the noise matrix needs to be constructed using
                    #cluster-neighborhoods noise model with treating some members of clusters as neighbours

                    #average over neighbours of first cluster which do no include the members of second cluster
                    averaged_matrices_cluster_i = average_over_some_qubits(matrices_cluster_i,
                                                                           dependencies_cluster_i,
                                                                           intersection_i)


                    #average over neighbours of second cluster which do no include the members of first cluster
                    averaged_matrices_cluster_j = average_over_some_qubits(matrices_cluster_j,
                                                                           dependencies_cluster_j,
                                                                           intersection_j)


                    qubits_indices_enumerated = dict(enumerate(cluster_i+cluster_j))
                    rev_map_enumerated = anf.get_enumerated_rev_map(qubits_indices_enumerated)

                    qubit_indices_for_construction = []
                    for clust in [cluster_i,cluster_j]:
                        qubit_indices_for_construction.append([rev_map_enumerated[ci] for ci in clust])

                    averaged_matrices_cluster_i['neighbours'] = [rev_map_enumerated[ci] for ci in intersection_i]
                    averaged_matrices_cluster_j['neighbours'] = [rev_map_enumerated[cj] for cj in intersection_j]

                    properly_formatted_lambdas = [averaged_matrices_cluster_i,averaged_matrices_cluster_j]


                    averaged_matrix_clusters_i_j = fa.create_big_lambda(properly_formatted_lambdas,qubit_indices_for_construction)



                    if calculate_mitigation_errors:
                        # qubits_clusters_ij = anf.get_enumerated_rev_map_from_indices(anf.lists_sum_multi([cluster_i,cluster_j]))
                        qubits_ij = anf.get_enumerated_rev_map_from_indices(anf.lists_sum_multi([cluster_i,cluster_j,dependencies_cluster_i,dependencies_cluster_j]))

                        clu_i, clu_j = [qubits_ij[qc] for qc in cluster_i],  [qubits_ij[qc] for qc in cluster_j],
                        deps_i, deps_j = [qubits_ij[qc] for qc in dependencies_cluster_i],  [qubits_ij[qc] for qc in dependencies_cluster_j]


                        all_qubit_ij = anf.lists_sum_multi([deps_i,deps_j])
                        all_qubits_outside_ij = list(set(all_qubit_ij).difference(set(clu_i+clu_j)))

                        # print(qubits_ij,dependencies_cluster_j)
                        # print(deps_j)
                        # print(all_qubits_outside_ij)
                        possible_states_ij_outside =  pdt.register_names_qubits(range(len(all_qubits_outside_ij)),len(all_qubits_outside_ij))

                        possible_states_ij = []
                        for k in range(len(possible_states_ij_outside)):
                            state_here = possible_states_ij_outside[k]
                            better_statesize= len(list(state_here))+len(clu_i)+len(clu_j)
                            new_state = np.zeros((better_statesize),dtype=str)

                            for ciiiii in clu_i:
                                new_state[ciiiii]='0'
                            for cjjjjj in clu_j:
                                new_state[cjjjjj] = '0'
                            for kurde in range(len(state_here)):
                                new_state[all_qubits_outside_ij[kurde]]=state_here[kurde]
                            new_state = ''.join([x for x in new_state])
                            possible_states_ij.append(new_state)




                        map_cluster = anf.get_enumerated_rev_map_from_indices(sorted(clu_i+clu_j))

                        matrices_cluster_i_proper, matrices_cluster_j_proper = copy.deepcopy(matrices_cluster_i), copy.deepcopy(matrices_cluster_j)
                        if len(deps_i)==0:
                            matrices_cluster_i_proper['neighbours'] = None
                        else:
                            matrices_cluster_i_proper['neighbours'] = deps_i

                        if len(deps_j) == 0:
                            matrices_cluster_j_proper['neighbours'] = None
                        else:
                            matrices_cluster_j_proper['neighbours'] = deps_j



                        mitigation_errors_ij = 0
                        for state_possible in possible_states_ij:
                            big_matrix_now = fa.create_big_lambda_modified([matrices_cluster_i_proper,matrices_cluster_j_proper],
                                                                       [clu_i,clu_j],
                                                                       state_possible,
                                                                       map_cluster
                                                                       )
                            err_now = np.linalg.norm(averaged_matrix_clusters_i_j - big_matrix_now, ord=1)
                            # print(err_now)
                            if err_now > mitigation_errors_ij:
                                mitigation_errors_ij = err_now


                        #TODO: Does this work already?



                        # anf.cool_print('Finish this code dude','')
                        # raise KeyboardInterrupt("finish this code dude")


            sorted_quest = True
            if cluster_i==cluster_j:
                pass
            else:
                for ccc1 in cluster_j:
                    for ccc0 in cluster_i:
                        if ccc0>ccc1:
                            sorted_quest = False


            whole_marginal = sorted(anf.lists_sum(cluster_i, cluster_j))
            if not sorted_quest:
                # TODO: swapping qubits, does it make sense?
                averaged_matrix = averaged_matrix_clusters_i_j
                import QREM.povmtools as qrem_pt
                # anf.ptr(averaged_matrix_clusters_i_j)

                qubits_in_here = cluster_i+cluster_j
                sorted_qubits_in_here = dict(enumerate(sorted(qubits_in_here)))
                print(cluster_i, cluster_j)
                rev_map = anf.get_enumerated_rev_map_from_indices(cluster_i+cluster_j)
                # print(qubits_in_here,sorted_qubits_in_here)

                qubits_in_here_dict = dict(enumerate(qubits_in_here))

                while qubits_in_here_dict!=sorted_qubits_in_here:
                    for index_qubit_hehe in range(len(qubits_in_here)-1):
                        if qubits_in_here[index_qubit_hehe]<qubits_in_here[index_qubit_hehe+1]:
                            pass
                        elif qubits_in_here[index_qubit_hehe+1]<qubits_in_here[index_qubit_hehe]:
                            averaged_matrix = qrem_pt.permute_matrix(averaged_matrix, len(whole_marginal),
                                                                     [index_qubit_hehe+1,index_qubit_hehe+2])

                            anf.cool_print('Swapping qubits:', )
                            print(qubits_in_here[index_qubit_hehe],qubits_in_here[index_qubit_hehe+1],index_qubit_hehe,index_qubit_hehe+1)

                            qubits_in_here[index_qubit_hehe], qubits_in_here[index_qubit_hehe + 1] = qubits_in_here[index_qubit_hehe + 1], qubits_in_here[index_qubit_hehe]




                    qubits_in_here_dict = dict(enumerate(qubits_in_here))



            else:
                averaged_matrix = averaged_matrix_clusters_i_j

            correction_matrix = np.linalg.inv(averaged_matrix)

            string_marginal = ''.join(['q%s' % qqq for qqq in whole_marginal])
            correction_indices[string_pair] = string_marginal
            correction_indices['q%s' % i] = string_marginal
            correction_indices['q%s' % j] = string_marginal

            noise_matrices[string_marginal] = averaged_matrix
            correction_matrices[string_marginal] = correction_matrix
            if calculate_mitigation_errors:
                # print(string_marginal, mitigation_errors_ij)
                mitigation_errors[string_marginal]= mitigation_errors_ij

                if mitigation_errors_ij/2>=0.04:
                    highly_correlated_qubits.append({'qubits':string_marginal,'error':mitigation_errors_ij/2})


    return correction_indices, correction_matrices, noise_matrices, highly_correlated_qubits, mitigation_errors



