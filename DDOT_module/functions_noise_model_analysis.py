"""
Created on 01.03.2021

@authors: Filip B. Maciejewski, Tomek Rybotycki, Oskar Słowik
@contact: filip.b.maciejewski@gmail.com
"""

import numpy as np
import re
import copy
import QREM




def get_averaged_noise_matrix_from_big_noise_matrix(big_lambda,
                                                    bits_of_interest):
    big_N = int(np.log2(big_lambda.shape[0]))
    small_N = len(bits_of_interest)
    normalization = 2 ** (big_N - small_N)

    classical_register_big = ["{0:b}".format(i).zfill(big_N) for i in range(2 ** big_N)]

    indices_small = {}

    for s in classical_register_big:
        small_string = ''.join([list(s)[b] for b in bits_of_interest])
        indices_small[s] = small_string

    small_lambda = np.zeros((2 ** small_N, 2 ** small_N))

    for i in range(2 ** big_N):
        for j in range(2 ** big_N):
            lambda_element = big_lambda[i, j]
            ideal_state = "{0:b}".format(j).zfill(big_N)
            measured_state = "{0:b}".format(i).zfill(big_N)

            ideal_state_small = indices_small[ideal_state]
            measured_state_small = indices_small[measured_state]

            small_lambda[int(measured_state_small, 2), int(ideal_state_small, 2)] += lambda_element / normalization
    return small_lambda



def get_small_noise_matrices_from_big_noise_matrix(big_lambda,
                                                  bits_of_interest,
                                                  neighbours_of_interest,
                                                  mapping=None):

    if len(neighbours_of_interest) == 0 or neighbours_of_interest is None:
        # print('no neighbours given, returning averaged matrix')
        small_lambdas = {'neighbours': None,
                        'lambdas': get_averaged_noise_matrix_from_big_noise_matrix(big_lambda,bits_of_interest)}
        return small_lambdas

    if len(np.setdiff1d(bits_of_interest,neighbours_of_interest))!=len(bits_of_interest):
        print(bits_of_interest,neighbours_of_interest)
        raise ValueError('Wrong indices')

    big_N = int(np.log2(big_lambda.shape[0]))
    small_N = len(bits_of_interest)
    neighbours_N = len(neighbours_of_interest)

    normalization = 2 ** (big_N - neighbours_N - small_N)

    classical_register_big = ["{0:b}".format(i).zfill(big_N) for i in range(2 ** big_N)]
    classical_register_neighbours = ["{0:b}".format(i).zfill(neighbours_N) for i in range(2 ** neighbours_N)]
    indices_small = {}

    for s in classical_register_big:
        small_string = ''.join([list(s)[b] for b in bits_of_interest])
        neighbours_string = ''.join([list(s)[b] for b in neighbours_of_interest])
        indices_small[s] = [small_string, neighbours_string]

    small_lambdas = {s: np.zeros((2 ** small_N, 2 ** small_N)) for s in classical_register_neighbours}
    for i in range(2 ** big_N):
        for j in range(2 ** big_N):
            lambda_element = big_lambda[i, j]
            ideal_state = "{0:b}".format(j).zfill(big_N)
            measured_state = "{0:b}".format(i).zfill(big_N)

            ideal_state_small = indices_small[ideal_state][0]
            measured_state_small = indices_small[measured_state][0]

            ideal_state_neighbours = indices_small[ideal_state][1]


            small_lambdas[ideal_state_neighbours][int(measured_state_small, 2), int(ideal_state_small, 2)] += lambda_element / normalization

    lambdas_dict = {'lambdas':small_lambdas,
                    }
    if mapping is None:
        lambdas_dict['neighbours'] = list(neighbours_of_interest)
    else:
        lambdas_dict['neighbours'] = [mapping[ni] for ni in neighbours_of_interest]

    return lambdas_dict

def calculate_correlations_pairs(marginal_noise_matrices,
                                 qubit_indices):

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






