"""
Created on 01.03.2021

@authors: Filip B. Maciejewski, Tomek Rybotycki, Oskar Słowik
@contact: filip.b.maciejewski@gmail.com
"""

import numpy as np
import re
import copy
import QREM
from QREM import povmtools
from QREM import ancillary_functions as anf


def get_averaged_noise_matrix_from_big_noise_matrix(big_lambda,
                                                    qubits_of_interest):
    """From high-dimensional noise matrix return lower-dimensional effective noise matrix acting on marginal averaged over all other qubits than "qubits_of_interest"
    :param big_lambda (array): array describing stochastic (classical) noise on the whole space
    :param qubits_of_interest (list of ints): list of integers labeling the qubits in marginal we are interested in

    :return: small_lambda (array): effective noise matrix acting on qubits from "qubits_of_interest" (averaged over all other qubits)
    """

    #What is the number of qubits in the whole space
    big_N = int(np.log2(big_lambda.shape[0]))

    #What is the number of qubits in marginal of interest
    small_N = len(qubits_of_interest)

    #Normalization when averaging over states of neighbours (each with the same probability)
    normalization = 2 ** (big_N - small_N)

    #classical register on all qubits
    classical_register_big = ["{0:b}".format(i).zfill(big_N) for i in range(2 ** big_N)]

    #create dictionary of the marginal states of qubits of interest for the whole register
    #(this function is storing data which could also be calculated in situ in the loops later)
    indices_small = {}
    for s in classical_register_big:
        small_string = ''.join([list(s)[b] for b in qubits_of_interest])
        indices_small[s] = small_string

    #prepare the noise matrix on "qubits_of_interest"
    small_lambda = np.zeros((2 ** small_N, 2 ** small_N))

    #go through all classical states
    for i in range(2 ** big_N):
        for j in range(2 ** big_N):
            lambda_element = big_lambda[i, j]

            #input state of all qubits in binary format
            input_state = "{0:b}".format(j).zfill(big_N)
            #measured state of all qubits in binary format
            measured_state = "{0:b}".format(i).zfill(big_N)

            #input state of marginal qubits in binary format
            input_state_small = indices_small[input_state]
            #measured state of all qubits in binary format
            measured_state_small = indices_small[measured_state]

            #element of small lambda labeled by (measured state | input state)
            small_lambda[int(measured_state_small, 2), int(input_state_small, 2)] += lambda_element

    #normalize matrix and return
    return small_lambda/normalization



def get_small_noise_matrices_from_big_noise_matrix(big_lambda,
                                                  qubits_of_interest,
                                                  neighbors_of_interest,
                                                  mapping=None):
    """From high-dimensional noise matrix return lower-dimensional effective noise matrices acting on "qubits_of_interest" conditioned on input states of "neighbors of interest"
    :param big_lambda (array): array describing stochastic (classical) noise on the whole space
    :param qubits_of_interest (list of ints): list of integers labeling the qubits in marginal we are interested in
    :param neighbors_of_interest (list of ints): list of integers labeling the qubits that affet noise matrix on "qubits_of_interest"

    :param (optional) mapping (dictionary): optional dictionary mapping qubits labels


    :return: lambdas_dict (dictionary):
             dictionary with two KEYS:
             lambdas_dict['lambdas'] = DICTIONARY, where KEY is the input state of "neighbors_of_interest", and VALUE is the noise matrix acting on "qubits_of_interested" for this fixed input state of neighbors
             lambdas_dict['neighbors'] = list of neighbors (this either copies param "neighbors_of_interest" or returns neighbors mapped according to "mapping" param)

    """

    #If there are no neighbors, then this corresponds to averaging over all qubits except "qubits_of_interest"
    if len(neighbors_of_interest) == 0 or neighbors_of_interest is None:
        small_lambdas = {'neighbours': None,
                        'lambdas': get_averaged_noise_matrix_from_big_noise_matrix(big_lambda,qubits_of_interest)}
        return small_lambdas


    #check if there is no collision between qubits of interest and neighbors of interest (if there is, then the model won't be consistent)
    if len(anf.lists_intersection(qubits_of_interest, neighbors_of_interest))!=0:
        print(qubits_of_interest,neighbors_of_interest)
        raise ValueError('Wrong indices')


    #What is the number of qubits in the whole space
    big_N = int(np.log2(big_lambda.shape[0]))

    #What is the number of qubits in marginal of interest
    small_N = len(qubits_of_interest)

    #What is the number of neighbors
    neighbours_N = len(neighbors_of_interest)

    #Normalization when averaging over states of non-neighbours (each with the same probability)
    normalization = 2 ** (big_N - neighbours_N - small_N)

    #classical register on all qubits
    classical_register_big = ["{0:b}".format(i).zfill(big_N) for i in range(2 ** big_N)]

    #classical register on neighbours
    classical_register_neighbours = ["{0:b}".format(i).zfill(neighbours_N) for i in range(2 ** neighbours_N)]

    #create dictionary of the marginal states of qubits of interest and neighbors for the whole register
    #(this function is storing data which could also be calculated in situ in the loops later)
    indices_small = {}
    for s in classical_register_big:
        small_string = ''.join([list(s)[b] for b in qubits_of_interest])
        neighbours_string = ''.join([list(s)[b] for b in neighbors_of_interest])
        #first place in list is label for state of "qubits_of_interest" and second for "neighbors_of_interest
        indices_small[s] = [small_string, neighbours_string]

    #initiate dictionary for which KEY is input state of neighbors and VALUE will the the corresponding noise matrix on "qubits_of_interest"
    small_lambdas = {s: np.zeros((2 ** small_N, 2 ** small_N)) for s in classical_register_neighbours}

    #go through all classical states
    for i in range(2 ** big_N):
        for j in range(2 ** big_N):
            lambda_element = big_lambda[i, j]

            #input state of all qubits in binary format
            input_state = "{0:b}".format(j).zfill(big_N)
            #measured state of all qubits in binary format
            measured_state = "{0:b}".format(i).zfill(big_N)

            #input state of "qubits_of_interest" in binary format
            input_state_small = indices_small[input_state][0]
            #measured state of "qubits_of_interest" in binary format
            measured_state_small = indices_small[measured_state][0]

            #input state of "neighbors_of_interest" in binary format
            input_state_neighbours = indices_small[input_state][1]

            #element of small lambda labeled by (measured state | input state), and the lambda itself is labeled by input state of neighbors
            small_lambdas[input_state_neighbours][int(measured_state_small, 2), int(input_state_small, 2)] += lambda_element

    #normalize matrices
    for s in classical_register_neighbours:
        small_lambdas[s]*=1/normalization

    #prepare dictionary with first key being the obtained noise matrices
    lambdas_dict = {'lambdas':small_lambdas,
                    }

    #add neighbors list to dictionary (either mapped or not)
    if mapping is None:
        lambdas_dict['neighbors'] = list(neighbors_of_interest)
    else:
        lambdas_dict['neighbors'] = [mapping[ni] for ni in neighbors_of_interest]

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



def get_neighborhood_treshold_statitsical_pairs(number_of_samples,
                                                number_of_qubits=1,
                                                probability_of_error=0.001):




    pairs_number = number_of_qubits*(number_of_qubits-1)/2
    eps1q = povmtools.get_statistical_error_bound(2,
                                                  number_of_samples,
                                                  probability_of_error,
                                                  pairs_number)


    return 2*eps1q




def get_clusters_and_neighbors_from_pairs(qubit_indices,
                                           correlations_2q,
                                           neighborhood_treshold=0.01,
                                           cluster_treshold=0.04):
    #TODO:



    clusters = {'q%s' % i: [] for i in qubit_indices}

    for qi in qubit_indices:
        for qj in qubit_indices:
            cij, cji = correlations_2q[qi, qj] , correlations_2q[qj, qi]
            if cij >= cluster_treshold or cji >= cluster_treshold:
                clusters['q%s' % qi].append(qj)


    clusters_list = []
    for qi in qubit_indices:
        clusters_list.append(sorted([qi]+clusters['q%s'%qi]))

    new_lists = copy.deepcopy(clusters_list)


    while anf.check_if_there_are_common_elements(new_lists):
        for i in range(len(new_lists)):
            cl0 = new_lists[i]
            for j in range(len(new_lists)):
                cl1 = new_lists[j]
                if len(anf.lists_intersection(cl0,cl1))!=0:
                    new_lists[i] = anf.lists_sum(cl0,cl1)

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
            clusters_dict['q%s'%qi] = sorted(anf.lists_difference(cluster,[qi]))

    return clusters_dict, neighborhoods_clusters, proper_clusters, neighborhoods_qubits



def cut_subset_sizes(clusters_neighbourhoods_dict,
                     correlations_table,
                     target_size = 5):

    cutted_dict = copy.deepcopy(clusters_neighbourhoods_dict)

    for cluster, neighbours in clusters_neighbourhoods_dict.items():
        correlations_now = []
        cluster_inds = anf.get_qubit_indices_from_string(cluster)
        for ni in neighbours:
            for ci in cluster_inds:
                correlations_now.append([ni,correlations_table[ci,ni]])

        sorted_correlations = sorted(correlations_now,key=lambda x: x[1],reverse=True)

        base_size = len(cluster_inds)

        cut_neighbourhood = []

        if base_size==target_size:
            pass
        else:
            for tup in sorted_correlations:
                if (base_size+len(cut_neighbourhood))==target_size:
                    # print()
                    break
                else:
                    ni = tup[0]
                    if ni not in cut_neighbourhood:
                        cut_neighbourhood.append(ni)

        cutted_dict[cluster] = sorted(cut_neighbourhood)


    return cutted_dict


