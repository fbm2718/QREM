"""
Created on 05.05.2021

@author: Filip Maciejewski
@contact: filip.b.maciejewski@gmail.com
"""

import os, pickle
import numpy as np
from QREM import ancillary_functions as anf
from QREM.DDOT_module.parent_classes.marginals_analyzer_base import MarginalsAnalyzerBase
from QREM.DDOT_module.child_classes.marginals_corrector import MarginalsCorrector
from QREM.DDOT_module.functions import functions_data_analysis as fda

module_directory = anf.get_module_directory()
tests_directory = module_directory + '/data_for_tests/'

# data used for testing
backend_name = 'ASPEN-8'
number_of_qubits = 23
date = '2020_05_04'

# specify whether count names are read from right to left (convention used by IBM)
reverse_counts = False

directory = tests_directory + 'DDOT/' + backend_name + '/N%s' % number_of_qubits + '/' + date + '/DDOT/'

files = os.listdir(directory)
with open(directory + files[-1], 'rb') as filein:
    dictionary_data = pickle.load(filein)

# in this exemplary file, the "true qubits" are labels of physical qubits
true_qubits = dictionary_data['true_qubits']

# in this example qubits will be labeled from 0 to len(true_qubits)-1
list_of_qubits = dictionary_data['list_of_qubits']
number_of_qubits = len(list_of_qubits)

# in this exemplary file, the "results_dictionary_preprocessed" key is pre-processed dictionary
# with experimental results that assumes qubits are labeled from 0 to len(true_qubits)-1

# the dictionary has the structure where:
# KEY = label for experiment (in DDOT, it is bitstring denoting INPUT state)
# VALUE = counts dictionary, where each KEY is bitstring denoting measurement outcome, and VALUE is
#        number of occurrences
dictionary_results = dictionary_data['results_dictionary_preprocessed']

# dictionary for which each KEY is classical INPUT state, and VALUE is dictionary of
# marginal distributions on all pairs of qubits
# in this example, we precomputed marginals for all experiments and all pairs of qubits
marginal_dictionaries_pairs = dictionary_data['marginals_dictionary_pairs']

# dictionary for which each KEY is label for qubits subset
# and VALUE is effective noise matrix on that subset
# in this example, we precomputed noise matrices for all pairs of qubits
noise_matrices_dictionary_pairs = dictionary_data['noise_matrices_dictionary_pairs']

correction_data_naive = dictionary_data['correction_data_naive']
correction_data_generator_no_clusters = dictionary_data['correction_data_no_clusters']

# data used for testing
date = '2020_05_04'
directory = tests_directory + 'DDOT/' + backend_name + '/N%s' % number_of_qubits + '/' + \
            date + '/ground_states/'
files = os.listdir(directory)
with open(directory + files[-1], 'rb') as filein:
    dictionary_data_ground_states = pickle.load(filein)

# print(dictionary_data_ground_states.keys())
results_dictionary_ground_states = dictionary_data_ground_states['dictionary_results_pre_processed']
hamiltonians_data_dictionary = dictionary_data_ground_states['hamiltonians_data_dictionary']

marginals_analyzer_noisy = MarginalsAnalyzerBase(results_dictionary=results_dictionary_ground_states,
                                                 reverse_counts=reverse_counts)

marginals_corrector_naive = MarginalsCorrector(
    experimental_results_dictionary=results_dictionary_ground_states,
    reverse_counts=reverse_counts,
    correction_data_dictionary=correction_data_naive
)

marginals_corrector_no_clusters = MarginalsCorrector(
    experimental_results_dictionary=results_dictionary_ground_states,
    reverse_counts=reverse_counts,
    correction_data_dictionary=correction_data_generator_no_clusters
)

from tqdm import tqdm

hamiltonians_data_keys = list(hamiltonians_data_dictionary.keys())

errors_noisy, errors_naive, errors_no_clusters = [], [], []

for h_index in tqdm(range(35, len(hamiltonians_data_keys))):
    key_hamiltonian = hamiltonians_data_keys[h_index]
    hamiltonian_data = hamiltonians_data_dictionary[key_hamiltonian]

    weights_dictionary = hamiltonian_data['weights_dictionary']
    energy_ideal = hamiltonian_data['ground_state_energy']

    # print(energy_ideal)

    keys_of_interest = list(weights_dictionary.keys())

    marginals_keys_naive = [marginals_corrector_naive.correction_indices[key] for key in
                            keys_of_interest]
    marginals_keys_no_clusters = [marginals_corrector_no_clusters.correction_indices[key] for key in
                                  keys_of_interest]

    marginals_keys_of_interest = anf.lists_sum_multi(
        [keys_of_interest, marginals_keys_naive, marginals_keys_no_clusters])

    marginal_subsets = [anf.get_qubit_indices_from_string(string_marginal) for string_marginal in
                        marginals_keys_of_interest]

    marginals_analyzer_noisy.compute_marginals(key_hamiltonian,
                                               marginal_subsets)

    # print(marg)
    marginals_dictionary_all = marginals_analyzer_noisy.marginals_dictionary[key_hamiltonian]

    # print(marginals_dictionary_all)
    energy_noisy = fda.estimate_energy_from_marginals(weights_dictionary=weights_dictionary,
                                                      marginals=marginals_dictionary_all)

    marginals_dictionary_naive = {}
    for key_naive in marginals_keys_naive:
        marginals_dictionary_naive[key_naive] = marginals_dictionary_all[key_naive]

    marginals_dictionary_no_clusters = {}
    for key_no_cluster in marginals_keys_no_clusters:
        marginals_dictionary_no_clusters[key_no_cluster] = marginals_dictionary_all[key_no_cluster]

    # correction_method = 'T_matrix'
    # method_kwargs = None

    # correction_method = 'hybrid_T_IBU'
    # method_kwargs = {'unphysicality_threshold': 0.05,
    #                  'iterations_number': 10,
    #                  'prior': None}

    correction_method = 'IBU'
    method_kwargs = {'iterations_number': 10,
                     'prior': None}

    # print(marginals_dictionary_naive)
    marginals_corrector_naive.correct_marginals(marginals_dictionary=marginals_dictionary_naive,
                                                method=correction_method,
                                                method_kwargs=method_kwargs)

    marginals_corrector_no_clusters.correct_marginals(
        marginals_dictionary=marginals_dictionary_no_clusters,
        method=correction_method,
        method_kwargs=method_kwargs)

    marginals_coarse_grained_corrected_naive = \
        marginals_corrector_naive.get_specific_marginals_from_marginals_dictionary(
            keys_of_interest,
            corrected=True)

    marginals_coarse_grained_corrected_no_clusters = \
        marginals_corrector_no_clusters.get_specific_marginals_from_marginals_dictionary(
            keys_of_interest,
            corrected=True)

    energy_corrected_naive = \
        fda.estimate_energy_from_marginals(weights_dictionary=weights_dictionary,
                                           marginals=
                                           marginals_coarse_grained_corrected_naive)

    energy_corrected__no_clusters = \
        fda.estimate_energy_from_marginals(
            weights_dictionary=weights_dictionary,
            marginals=
            marginals_coarse_grained_corrected_no_clusters)

    print(energy_ideal, energy_noisy, energy_corrected_naive, energy_corrected__no_clusters)

    err_noisy = abs(energy_ideal - energy_noisy) / number_of_qubits
    err_corrected_naive = abs(energy_ideal - energy_corrected_naive) / number_of_qubits
    err_corrected_no_clusters = abs(energy_ideal - energy_corrected__no_clusters) / number_of_qubits

    errors_noisy.append(err_noisy)
    errors_naive.append(err_corrected_naive)
    errors_no_clusters.append(err_corrected_no_clusters)
    raise KeyError

# noisy
mean_noisy = np.mean(errors_noisy)
std_noisy = np.std(errors_noisy)

# corrected - naive
mean_naive = np.mean(errors_naive)
std_naive = np.std(errors_naive)

# corrected - no clusters
mean_no_clusters = np.mean(errors_no_clusters)
std_no_clusters = np.std(errors_no_clusters)

anf.cool_print('Mean error corrected (naive):',
               np.round(mean_naive, anf.find_significant_digit(std_naive) + 1))
anf.cool_print('Mean error corrected (no clusters):',
               np.round(mean_no_clusters, anf.find_significant_digit(std_no_clusters) + 1))
anf.cool_print('Ratio of means no clusters:',
               np.round(mean_no_clusters / mean_naive, anf.find_significant_digit(std_naive) + 1))
anf.cool_print('Mean error noisy:', np.round(mean_noisy, anf.find_significant_digit(std_noisy) + 1))
anf.cool_print('Ratio of means noisy:',
               np.round(mean_noisy / mean_naive, anf.find_significant_digit(std_naive) + 1))
