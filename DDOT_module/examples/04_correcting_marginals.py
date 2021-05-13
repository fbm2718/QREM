"""
Created on 05.05.2021

@author: Filip Maciejewski
@contact: filip.b.maciejewski@gmail.com
"""

import os, pickle
import numpy as np
from tqdm import tqdm
from QREM import ancillary_functions as anf
from QREM.DDOT_module.parent_classes.marginals_analyzer_base import MarginalsAnalyzerBase
from QREM.DDOT_module.child_classes.marginals_corrector import MarginalsCorrector
from QREM.DDOT_module.functions import functions_data_analysis as fda


module_directory = anf.get_module_directory()
tests_directory = module_directory + '/data_for_tests/'

# data used for testing
backend_name = 'ibmq_16_melbourne'
number_of_qubits = 15
date = '2020_05_07'

name_of_hamiltonian = '2SAT'

# specify whether count names are read from right to left (convention used by IBM)
if backend_name == 'ibmq_16_melbourne':
    reverse_counts = True
elif backend_name == 'ASPEN-8':
    reverse_counts = False
else:
    raise ValueError('Wrong backend')

################## GET NOISE CHARACTERIZATION DATA ##################

directory = tests_directory + 'mitigation_on_marginals/' + backend_name + \
            '/N%s' % number_of_qubits + '/' + date + '/DDOT/'

files = os.listdir(directory)
with open(directory + '04_test_results_correction_data.pkl', 'rb') as filein:
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
marginal_dictionaries_pairs = dictionary_data['marginals_dictionary']

# dictionary for which each KEY is label for qubits subset
# and VALUE is effective noise matrix on that subset
# in this example, we precomputed noise matrices for all pairs of qubits
noise_matrices_dictionary_pairs = dictionary_data['noise_matrices_dictionary']

# get saved correction data
correction_data = dictionary_data['correction_data']

################## GET RESULTS OF BENCHMARKS ##################
date = '2020_05_07'
directory = tests_directory + 'mitigation_on_marginals/' + backend_name + '/N%s' % number_of_qubits + '/' + \
            date + '/ground_states/'

with open(directory + '00_test_results_' + name_of_hamiltonian + '.pkl', 'rb') as filein:
    dictionary_data_ground_states = pickle.load(filein)

results_dictionary_ground_states = dictionary_data_ground_states['dictionary_results_pre_processed']
hamiltonians_data_dictionary = dictionary_data_ground_states['hamiltonians_data_dictionary']
hamiltonians_data_keys = list(results_dictionary_ground_states.keys())

################## ESTIMATE ENERGIES OF LOCAL HAMILTONIANS ##################

# Get instance of marginals analyzer to estimate noisy marginals (which will be later corrected)
marginals_analyzer_noisy = MarginalsAnalyzerBase(results_dictionary=results_dictionary_ground_states,
                                                 bitstrings_right_to_left=reverse_counts)

# Get instance of marginals corrector which will be used, well, to correct marginals
marginals_corrector = MarginalsCorrector(
    experimental_results_dictionary=results_dictionary_ground_states,
    bitstrings_right_to_left=reverse_counts,
    correction_data_dictionary=correction_data
)

errors_noisy, errors_corrected = [], []
ratios_naive = []

perfect_results = 0

# Choose the noise mitigation method implemented on marginals
correction_method = 'T_matrix'
method_kwargs = None

# correction_method = 'hybrid_T_IBU'
# method_kwargs = {'unphysicality_threshold': 0.05,
#                  'iterations_number': 10,
#                  'prior': None}

# correction_method = 'T_matrix'
# method_kwargs = {'iterations_number': 10,
#                  'prior': None}


for h_index in tqdm(range(0, len(hamiltonians_data_keys))):
    key_hamiltonian = hamiltonians_data_keys[h_index]
    hamiltonian_data = hamiltonians_data_dictionary[key_hamiltonian]

    # The needed Hamiltonian data includes weigths dictionary
    # The KEYS of this dictionary are labels for qubits' subsets
    # and VALUES are corresponding coefficients in local terms of Hamiltonian
    weights_dictionary = hamiltonian_data['weights_dictionary']

    # This the energy that should be obtained in theory
    energy_ideal = hamiltonian_data['ground_state_energy']

    # We take the labels of marginals that we want to estimate
    # (needed to estimate expected value of our Hamiltonian)
    marginals_labels_hamiltonian = list(weights_dictionary.keys())

    # Here we take the labels for marginals that are needed to be corrected in order to
    # obtain the marginals we are interested in.
    # This is in general not the same as "marginals_labels_hamiltonian" because if the qubits
    # are highly correlated it is preferable to first correct the bigger qubit subset (clusters)
    # and than coarse-grain it to obtain marginal of interest
    marginals_labels_correction = [marginals_corrector.correction_indices[key] for key in
                                   marginals_labels_hamiltonian]

    # Take all labels of marginals - we want to estimate them all
    marginals_labels_all = anf.lists_sum_multi(
        [marginals_labels_hamiltonian, marginals_labels_correction])

    # Get subset of qubits on which marginals are defined
    marginal_subsets = [anf.get_qubit_indices_from_string(string_marginal) for string_marginal in
                        marginals_labels_all]

    # Calculate all of the marginals needed
    marginals_analyzer_noisy.compute_marginals(key_hamiltonian,
                                               marginal_subsets)

    # Get dictionary with computed marginals
    marginals_dictionary_all = marginals_analyzer_noisy.marginals_dictionary[key_hamiltonian]

    # Estimate energy from noisy marginals
    energy_noisy = fda.estimate_energy_from_marginals(weights_dictionary=weights_dictionary,
                                                      marginals=marginals_dictionary_all)

    marginals_dictionary_to_correct = {}
    for key_naive in marginals_labels_correction:
        marginals_dictionary_to_correct[key_naive] = marginals_dictionary_all[key_naive]

    # Correct marginals
    marginals_corrector.correct_marginals(marginals_dictionary=marginals_dictionary_to_correct,
                                          method=correction_method,
                                          method_kwargs=method_kwargs)

    # Coarse-grain some of the corrected marginals to obtain the ones that appear in Hamiltonian
    marginals_coarse_grained_corrected = \
        marginals_corrector.get_specific_marginals_from_marginals_dictionary(
            marginals_labels_hamiltonian,
            corrected=True)

    # Estimate energy from corrected marginals
    energy_corrected = \
        fda.estimate_energy_from_marginals(weights_dictionary=weights_dictionary,
                                           marginals=
                                           marginals_coarse_grained_corrected)

    # Get error per qubit
    err_noisy = abs(energy_ideal - energy_noisy) / number_of_qubits
    err_corrected_naive = abs(energy_ideal - energy_corrected) / number_of_qubits

    errors_noisy.append(err_noisy)
    errors_corrected.append(err_corrected_naive)



################## PRINT RESULTS ##################
all_errors = [errors_noisy, errors_corrected]
all_ratios = [ratios_naive]

words = ['noisy', 'naive']

dictionary_final = {}
for i in range(len(all_errors)):
    key_now, list_now = words[i], all_errors[i]

    dictionary_final[key_now + '_mean_error'] = np.round(np.mean(list_now),
                                                         anf.find_significant_digit(
                                                             np.std(list_now)) + 1)

for j in [0]:
    k = j
    if j > 1:
        k = j - 1
    key_now, list_now = words[j], all_ratios[k]

    dictionary_final[key_now + '_mean_ratio'] = np.round(np.mean(list_now),
                                                         anf.find_significant_digit(
                                                             np.std(list_now)) + 1)

for l in [0]:
    key_now, list_now = words[l], all_errors[l]

    dictionary_final[key_now + '_ratio_of_means'] = np.round(
        np.mean(list_now) / np.mean(errors_corrected),
        anf.find_significant_digit(
            np.std(list_now)) + 1)

counter = 0
for key, value in dictionary_final.items():
    anf.cool_print(key, value)
    if counter in [2]:
        print()
    counter += 1

anf.cool_print('Number of perfect results:', perfect_results)
