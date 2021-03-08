"""
Created on 08.03.2021

@authors: Filip B. Maciejewski, Tomek Rybotycki, Oskar Słowik
@contact: filip.b.maciejewski@gmail.com
"""

import numpy as np
import re
import copy
import QREM


def get_qubit_indices_from_string(qubits_string,
                                  with_q=False):

    """Return list of qubit indices from the string of the form "q0q1q22q31"
    :param qubits_string (string): string which has the form of "q" followed by qubit index
    :param (optional) with_q (Boolean): specify whether returned indices should be in form of string with letter

    :return: list of qubit indices:

    depending on value of parameter "with_q" the mapping will be one of the following:

    if with_q:
        'q1q5q13' -> ['q1','q5','q13']
    else:
        'q1q5q13' -> [1,5,13]

    """

    numbers = re.findall(r'\d+', qubits_string)

    if with_q:
        qubits = ['q'+s for s in numbers]
    else:
        qubits = [int(s) for s in numbers]

    return qubits




def get_state_from_circuit_name(circuit_name):
    state_name = ''
    for string in circuit_name:
        if string in ['1','X','x']:
            state_name+='1'
        elif string in ['0','I','i','id','Id']:
            state_name+='0'

    return state_name


def get_mini_dict(number_of_qubits):
    register = QREM.povmtools.register_names_qubits(range(number_of_qubits),number_of_qubits)
    return {key:np.zeros((int(2**number_of_qubits),1)) for key in register}




def get_marginals_from_counts_dict(counts,
                                   marginals_dictionary,
                                   reverse_counts = False,
                                   normalize = False,
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
        qubits_indices = get_qubit_indices_from_string(qubits_string)
        if qubits_mapping is not None:
            bits_of_interest = [qubits_mapping(qubits_indices[i]) for i in range(qubits_indices)]
        else:
            bits_of_interest = qubits_indices

        for count, ticks in counts.items():
            if reverse_counts:
                count = count[::-1]

            marginal_key_now = ''.join([count[b] for b in bits_of_interest])
            marginals_dictionary[qubits_string][int(marginal_key_now,2)]+=ticks

    if normalize:
        for qubits_string in marginals_dictionary.keys():
            marginals_dictionary[qubits_string]*=1/sum(marginals_dictionary[qubits_string])

    return marginals_dictionary





def get_subsets_marginals_from_counts(results_dictionary,
                              subsets,
                              reverse_counts,
                              qubits_mapping = None):


    """Return dictionary of marginal probability distributions from results_dictionary
    :param results_dictionary (dictionary): results dictionary for which KEY is the bitstring denoting INPUT CLASSICAL STATE, while VALUE is the counts dictionary with results of the experiments
    :param subsets (list of lists of ints): list of lists. Each list contains labels of subset of qubits for which marginal distributions are to be calculated.
   es
    :param reverse_counts (Boolean): Specify whether measurement result bitsstring should be reversed before adding to marginal (this is the case for qiskit where bits are counted from right)
    :param (optional) qubits_mapping (dict): optional dictionary with qubits labels mapping


    :return: marginals_dictionary (dictionary): the dictionary for which each KEY is the same as for "results_dictionary" and each VALUE is the dictionary of marginals as returned by function "get_marginals_from_counts_dict"

    """

    marginals_dictionaries_template = {'q'.join(sub):np.zeros((2**(int(len(sub),1)))) for sub in subsets}

    marginal_dictionaries = {}

    for what_we_put, counts in results_dictionary.items():
        marginals_dict_now = get_marginals_from_counts_dict(counts,
                                   copy.deepcopy(marginals_dictionaries_template),
                                   reverse_counts = reverse_counts,
                                   normalize = False,
                                   qubits_mapping = qubits_mapping)

        marginal_dictionaries[what_we_put] = marginals_dict_now

    return marginal_dictionaries




def get_noise_matrix_from_counts_dict(counts_dict,
                                      number_of_qubits=None):
    if number_of_qubits is None:
        number_of_qubits = len(list(counts_dict.keys())[0])
    lam = np.zeros((2**number_of_qubits,2**number_of_qubits))
    for key, val in counts_dict.items():
        lam[:,int(key,2)] = val[:,0]
    return lam





def get_subset_noise_matrices_from_marginals(marginal_dictionaries,
                                            subsets,
                                            max_subset_length=7):


    marginals_dictionaries_template = {'q'.join(sub):np.zeros((2**(int(len(sub),1)))) for sub in subsets}

    mini_dicts_template = {i + 1: get_mini_dict(i + 1) for i in range(max_subset_length)}


    marginal_dictionaries_subsets = {}
    for key_marginal in marginals_dictionaries_template.keys():
        qubits_now = get_qubit_indices_from_string(key_marginal)
        mini_dict_now = copy.deepcopy(mini_dicts_template[len(qubits_now)])


        for what_we_put, dictionary_marginal in marginal_dictionaries.items():
            input_marginal = ''.join([what_we_put[x] for x in qubits_now])
            mini_dict_now[input_marginal]+=dictionary_marginal[key_marginal]

        for key_small in mini_dict_now.keys():
            mini_dict_now[key_small] *= 1 / np.sum(mini_dict_now[key_small])

        marginal_dictionaries_subsets[key_marginal] = mini_dict_now

    noise_matrices = {}

    for marginal_key, marginal_dict in marginal_dictionaries_subsets.items():
        noise_matrices[marginal_key] = get_noise_matrix_from_counts_dict(marginal_dict)

    return noise_matrices



