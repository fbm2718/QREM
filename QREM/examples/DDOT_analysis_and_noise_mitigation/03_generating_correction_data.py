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

from functions import ancillary_functions as anf
from QREM.noise_mitigation.CorrectionDataGenerator import \
    CorrectionDataGenerator


"""
Examples here analyze data obtained in experiments described in [0.5].

Please see examples/DDOT_implementation/ to create and implement new experiments.

Examples 01, 02 and 03 are all merged together in example 04, which can be a starting point.
"""

module_directory = anf.get_module_directory()
tests_directory = module_directory + '/saved_data/'

# data used for testing
backend_name = 'ibmq_16_melbourne'

# Specify whether save calculated data
saving = True

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

directory = tests_directory + 'mitigation_on_marginals/' + backend_name \
            + '/number_of_qubits_%s' % number_of_qubits + '/' + date + '/DDOT/'

files = os.listdir(directory)
with open(directory + '03_test_results_noise_models.pkl', 'rb') as filein:
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
marginal_dictionaries_initial = dictionary_data['marginals_dictionary']

# dictionary for which each KEY is label for qubits subset
# and VALUE is effective noise matrix on that subset
# in this example, we precomputed noise matrices for all pairs of qubits
noise_matrices_dictionary_initial = dictionary_data['noise_matrices_dictionary']

# Get clusters and neighbors from saved data
clusters_list, neighborhoods = dictionary_data['clusters_labels_list'], dictionary_data[
    'neighborhoods']

# Get instance of correction data generator - it will generate correction matrices for marginals_dictionary
# based on provided noise model
correction_data_generator = CorrectionDataGenerator(results_dictionary_ddt=dictionary_results,
                                                    bitstrings_right_to_left=bitstrings_right_to_left,
                                                    number_of_qubits=number_of_qubits,
                                                    marginals_dictionary=
                                                    marginal_dictionaries_initial,
                                                    clusters_list=clusters_list,
                                                    neighborhoods=neighborhoods,
                                                    noise_matrices_dictionary=
                                                    noise_matrices_dictionary_initial)

all_pairs = [[qi, qj] for qi in list_of_qubits for qj in list_of_qubits if qj > qi]

# Get data needed to make corrections for two-qubit marginals_dictionary (for 2-local Hamiltonian problems)
correction_data = correction_data_generator.get_pairs_correction_data(all_pairs,
                                                                      show_progress_bar
                                                                      =True)

if saving:
    # Update dictionary to be saved
    dictionary_data['correction_data'] = correction_data

    # Save results
    directory = tests_directory + 'mitigation_on_marginals/' + backend_name \
                + '/number_of_qubits_%s' % number_of_qubits + '/' + date + '/DDOT/'

    anf.save_results_pickle(dictionary_data,
                            directory,
                            '04_test_results_correction_data')
