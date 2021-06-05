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

from QREM.functions import ancillary_functions as anf
from QREM.noise_characterization.tomography.DDTMarginalsAnalyzer import DDTMarginalsAnalyzer


"""
Examples here analyze data obtained in experiments described in [0.5].

Please see examples/DDOT_implementation/ to create and implement new experiments.

Examples 01, 02 and 03 are all merged together in example 04, which can be a starting point.
"""

module_directory = anf.get_module_directory()
tests_directory = module_directory + '/saved_data/'

# specify data used for testing
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
# print(files)
with open(directory + '00_test_results_counts.pkl', 'rb') as filein:
    dictionary_data = pickle.load(filein)

# in this exemplary file, the "true qubits" are labels of physical qubits
true_qubits = dictionary_data['true_qubits']

number_of_qubits = len(true_qubits)

# qubits will be labeled from 0 to len(true_qubits)-1
list_of_qubits = range(number_of_qubits)

# in this exemplary file, the "results_dictionary_preprocessed" key is pre-processed dictionary
# with experimental results that assumes qubits are labeled from 0 to len(true_qubits)-1

# the dictionary has the structure where:
# KEY = label for experiment (in DDOT, it is bitstring denoting INPUT state)
# VALUE = counts dictionary, where each KEY is bitstring denoting measurement outcome, and VALUE is
#        number of occurrences
dictionary_results = dictionary_data['results_dictionary_preprocessed']

# get instance of marginals_dictionary analyzer for ddot experiments
marginals_analyzer_ddot = DDTMarginalsAnalyzer(dictionary_results,
                                               bitstrings_right_to_left)

# get indices of all qubit pairs (in ascending order)
all_pairs = [[i, j] for i in list_of_qubits for j in list_of_qubits if j > i]

# compute marginal distributions for all experiments and all qubit pairs
# showing progress bar requires 'tqdm' package
marginals_analyzer_ddot.compute_all_marginals(all_pairs,
                                              show_progress_bar=True)

if saving:
    # Save marginals_dictionary
    directory = tests_directory + 'mitigation_on_marginals/' + backend_name + \
                '/number_of_qubits_%s' % number_of_qubits + '/' + date + '/DDOT/'
    dictionary_data['marginals_dictionary_pairs'] = marginals_analyzer_ddot.marginals_dictionary

    anf.save_results_pickle(dictionary_data, directory, '01_test_results_marginals_pairs')

# compute average noise matrices on all qubits pairs;
# this will be used for initial noise analysis
# showing progress bar requires 'tqdm' package
marginals_analyzer_ddot.compute_subset_noise_matrices_averaged(all_pairs,
                                                               show_progress_bar=True)

if saving:
    # Save averaged noise matrices
    dictionary_data['noise_matrices_dictionary_pairs'] = \
        marginals_analyzer_ddot.noise_matrices_dictionary

    anf.save_results_pickle(dictionary_data,
                            directory,
                            '02_test_results_noise_matrices_pairs')
