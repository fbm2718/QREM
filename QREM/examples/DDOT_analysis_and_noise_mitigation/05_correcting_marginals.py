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

import os
import pickle
from tqdm import tqdm
import numpy as np
from QREM.noise_mitigation.MarginalsCorrector import MarginalsCorrector
from QREM.functions import functions_data_analysis as fda, ancillary_functions as anf

"""
Examples here analyze data obtained in experiments described in [0.5].

Please see examples/DDOT_implementation/ to create and implement new experiments.
"""

module_directory = anf.get_module_directory()
tests_directory = module_directory + '/saved_data/'

# data used for testing
backend_name = 'ibmq_16_melbourne'

# Specify whether save calculated data
saving = True

# This is name of Hamiltonians for which ground states were implemented to benchmark error-mitigation
name_of_hamiltonian = '2SAT'

# specify whether count names are read from right to left (convention used by IBM)
if backend_name == 'ibmq_16_melbourne':
    date = '2020_10_12'
    number_of_qubits = 15
    bitstrings_right_to_left = True
elif backend_name == 'ASPEN-8':
    date = '2020_12_31'
    number_of_qubits = 23
    bitstrings_right_to_left = False
else:
    raise ValueError('Wrong backend name')

################## GET NOISE CHARACTERIZATION DATA ##################
directory = tests_directory + 'mitigation_on_marginals/' + backend_name + \
            '/number_of_qubits_%s' % number_of_qubits + '/' + date + '/DDOT/'

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
# in this example, we precomputed marginals_dictionary for all experiments and all pairs of qubits
marginal_dictionaries_pairs = dictionary_data['marginals_dictionary']

# dictionary for which each KEY is label for qubits subset
# and VALUE is effective noise matrix on that subset
# in this example, we precomputed noise matrices for all pairs of qubits
noise_matrices_dictionary_pairs = dictionary_data['noise_matrices_dictionary']

# get saved correction data
correction_data = dictionary_data['correction_data']

'''
............WORKFLOW............

I. POST-PROCESSING -- using created noise model and correction data 
                      to mitigate noise in experimental results.

Input: 
potentially_stochastic_matrix) correction data obtained from DDT experiments (above)
b) results of experiments we wish to correct on the level of marginals_dictionary 

0. Get instance of MarginalsCorrector object.
1. Get marginals_dictionary from experimental results - the ones needed for your experiment, and the ones 
                                             needed to correct it (see descriptions in the code).
2. Correct marginals_dictionary.
'''

################## GET RESULTS OF BENCHMARKS ##################
'''
The benchmarks here are the following: 
1. Take random instance of  2-local, diagonal Hamiltonians (such as those occurring in QAOA).
2. Implement ground state of that Hamiltonian (note that those are classical states, i_index.e., states
from computational basis).
3. Estimate the energy of that ground state and compare it to ideal value:
with and without error-mitigation.

NOTE: As this is just benchmark for error-mitigation, we know the answer beforehand, so the Hamiltonian
is earlier solved on classical computer.
'''

directory = tests_directory + 'mitigation_on_marginals/' + backend_name \
            + '/number_of_qubits_%s' % number_of_qubits + '/' + date + '/ground_states/'

with open(directory + '00_test_results_' + name_of_hamiltonian + '.pkl', 'rb') as filein:
    dictionary_data_ground_states = pickle.load(filein)

# Get data which will be needed for error-mitigation
results_dictionary_ground_states = dictionary_data_ground_states['dictionary_results_pre_processed']
hamiltonians_data_dictionary = dictionary_data_ground_states['hamiltonians_data_dictionary']
hamiltonians_data_keys = list(results_dictionary_ground_states.keys())



# Due to some error, in the saved data from Ref. [0.5] there is one additional Hamiltonian.
if backend_name == 'ASPEN-8':
    hamiltonians_data_keys.remove('00111100011111111001000')

################## ESTIMATE ENERGIES OF LOCAL HAMILTONIANS ##################

# Get instance of marginals_dictionary analyzer to estimate noisy marginals_dictionary (which will be later corrected)
# marginals_analyzer_noisy = MarginalsAnalyzerBase(results_dictionary=results_dictionary_ground_states,
#                                                  bitstrings_right_to_left=bitstrings_right_to_left)

# Get instance of marginals_dictionary corrector which will be used, well, to correct marginals_dictionary
marginals_corrector = MarginalsCorrector(
    experimental_results_dictionary=results_dictionary_ground_states,
    bitstrings_right_to_left=bitstrings_right_to_left,
    correction_data_dictionary=correction_data
)

errors_noisy, errors_corrected = [], []

# Choose the noise mitigation method_name implemented on marginals_dictionary
correction_method = 'T_matrix'
method_kwargs = {'ensure_physicality': True}

# Go through all Hamiltonians
for h_index in tqdm(range(0, len(hamiltonians_data_keys))):
    key_hamiltonian = hamiltonians_data_keys[h_index]
    hamiltonian_data = hamiltonians_data_dictionary[key_hamiltonian]

    # The needed Hamiltonian data includes weights dictionary
    # The KEYS of this dictionary are labels for qubits' subsets_list
    # and VALUES are corresponding coefficients in local terms of Hamiltonian
    weights_dictionary = hamiltonian_data['weights_dictionary']

    # This the energy that should be obtained in theory
    energy_ideal = hamiltonian_data['ground_state_energy']

    # We take the labels of marginals_dictionary that we want to estimate
    # (needed to estimate expected value of our Hamiltonian)
    marginals_labels_hamiltonian = list(weights_dictionary.keys())

    # Here we take the labels for marginals_dictionary that are needed to be corrected in order to
    # obtain the marginals_dictionary we are interested in.
    # This is in general NOT the same as "marginals_labels_hamiltonian" because if the qubits
    # are highly correlated it is preferable to first correct the bigger qubit subset (clusters)
    # and than coarse-grain it to obtain marginal of interest
    marginals_labels_correction = [marginals_corrector.correction_indices[key] for key in
                                   marginals_labels_hamiltonian]

    # Take all labels of marginals_dictionary - we want to estimate them all
    marginals_labels_all = anf.lists_sum_multi(
        [marginals_labels_hamiltonian, marginals_labels_correction])

    # Get subset of qubits on which marginals_dictionary are defined
    marginal_subsets = [anf.get_qubit_indices_from_string(string_marginal) for string_marginal in
                        marginals_labels_all]

    # Calculate all of the marginals_dictionary needed
    marginals_corrector.compute_marginals(key_hamiltonian,
                                          marginal_subsets)

    # Get dictionary with computed marginals_dictionary
    marginals_dictionary_all = marginals_corrector.marginals_dictionary[key_hamiltonian]

    # Estimate energy from noisy marginals_dictionary
    energy_noisy = fda.estimate_energy_from_marginals(weights_dictionary=weights_dictionary,
                                                      marginals_dictionary=marginals_dictionary_all)

    # Create dictionary with marginals_dictionary to-be-corrected
    marginals_dictionary_to_correct = {}
    for key_naive in marginals_labels_correction:
        marginals_dictionary_to_correct[key_naive] = marginals_dictionary_all[key_naive]

    # Correct marginals_dictionary
    marginals_corrector.correct_marginals(marginals_dictionary=marginals_dictionary_to_correct,
                                          method=correction_method,
                                          method_kwargs=method_kwargs)
    # Coarse-grain some of the corrected marginals_dictionary to obtain the ones that appear in Hamiltonian
    marginals_coarse_grained_corrected = \
        marginals_corrector.get_specific_marginals_from_marginals_dictionary(
            marginals_labels_hamiltonian,
            corrected=True)

    # Estimate energy from corrected marginals_dictionary
    energy_corrected = \
        fda.estimate_energy_from_marginals(weights_dictionary=weights_dictionary,
                                           marginals_dictionary=
                                           marginals_coarse_grained_corrected)

    # Get error per qubit
    err_noisy = abs(energy_ideal - energy_noisy) / number_of_qubits
    err_corrected_naive = abs(energy_ideal - energy_corrected) / number_of_qubits

    errors_noisy.append(err_noisy)
    errors_corrected.append(err_corrected_naive)

################## PRINT RESULTS ##################
all_errors = [errors_noisy, errors_corrected]

words = ['noisy', 'corrected']

dictionary_to_print = {}
for i_index in range(len(all_errors)):
    key_now, list_now = words[i_index], all_errors[i_index]

    dictionary_to_print[key_now + '_mean_error'] = np.round(np.mean(list_now),
                                                            anf.find_significant_digit(
                                                                np.std(list_now)) + 1)

for l_index in [0]:
    key_now, list_now = words[l_index], all_errors[l_index]

    dictionary_to_print[key_now + '_ratio_of_means'] = np.round(
        np.mean(list_now) / np.mean(errors_corrected),
        anf.find_significant_digit(
            np.std(list_now)) + 1)

counter = 0
for key, value in dictionary_to_print.items():
    anf.cool_print(key, value)
    if counter in [2]:
        print()
    counter += 1

if saving:
    # Update dictionary to be saved
    dictionary_data['corrected_data'] = {'errors_noisy': errors_noisy,
                                         'errors_corrected': errors_corrected}

    # Save results
    directory = tests_directory + 'mitigation_on_marginals/' + backend_name \
                + '/number_of_qubits_%s' % number_of_qubits + '/' + date + '/DDOT/'

    anf.save_results_pickle(dictionary_data,
                            directory,
                            '05_test_results_corrected_data_' + name_of_hamiltonian)
