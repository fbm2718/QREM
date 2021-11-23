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
from QREM.noise_mitigation.CorrectionDataGenerator import \
    CorrectionDataGenerator

"""
Examples here analyze data obtained in experiments described in [0.5].

Please see examples/DDOT_implementation/ to create and implement new experiments.

Examples 01, 02 and 03 are all merged together in example 04 (here), which can be a starting point.
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

'''
............WORKFLOW............

I. PRE-PROCESSING -- getting noise model and correction data.

Input: results of Diagonal Detector Tomography Data (DDT). 

0. Get instance of CorrectionDataGenerator object.
1. Calculate marginals_dictionary on pairs of qubits.
2. Based on those marginals_dictionary, construct noise model for faulty detector.
3. Based on that noise model, create correction matrices for marginals_dictionary. 

'''

# 0.
# We get instance of Correction Data Generator
correction_data_generator = CorrectionDataGenerator(results_dictionary_ddt=dictionary_results,
                                                    bitstrings_right_to_left=bitstrings_right_to_left,
                                                    number_of_qubits=number_of_qubits)

# we define list of subsystems we wish to first calculate marginals_dictionary for
all_pairs = [[qi, qj] for qi in list_of_qubits for qj in list_of_qubits if qj > qi]

# 1.
# compute marginal distributions for all experiments and all qubit pairs


anf.cool_print('Getting marginals from DDOT experiments', '...', 'green')
correction_data_generator.compute_all_marginals(all_pairs,
                                                show_progress_bar=True)
anf.cool_print('DONE', '\n', 'green')

# 1.
# compute average noise matrices on all qubits pairs;
# this will be used for initial noise analysis
anf.cool_print('Getting averaged noise matrices on pairs of qubits', '...', 'green')
correction_data_generator.compute_subset_noise_matrices_averaged(all_pairs,
                                                                 show_progress_bar=True)
anf.cool_print('DONE', '\n', 'green')
# 2.
# compute correlations table for qubit pairs
correction_data_generator.compute_correlations_table_pairs()
"""
Correlations are defined as:

c_{j -> i_index} = 1/2 * || Lambda_{i_index}^{Y_j = '0'} - Lambda_{i_index}^{Y_j = '0'}||_{l1}

Where Lambda_{i_index}^{Y_j} is an effective noise matrix on qubit "i_index" (averaged over all other 
of qubits except "j"), provided that input state of qubit "j" was "Y_j". Hence, c_{j -> i_index}
measures how much noise on qubit "i_index" depends on the input state of qubit "j".
"""

if backend_name == 'ibmq_16_melbourne':
    # those are values used in Ref. [0.5]
    threshold_clusters = 0.04
    threshold_neighbors = 0.01
elif backend_name == 'ASPEN-8':
    # those are values used in Ref. [0.5]
    threshold_clusters = 0.06
    threshold_neighbors = 0.02
else:
    raise ValueError('Wrong SDK_name')

# set maximal size of cluster+neighborhood set
maximal_size = 5

# Choose clustering method_name and its kwargs
# NOTE: see descriptions of the relevant functions for available options
# NOTE 2: this is method_name used in Ref. [0.5]
clustering_method = 'pairwise'
clustering_function_arguments = {'cluster_threshold': threshold_clusters}

anf.cool_print('Constructing noise model', '...', 'green')
# compute clusters based on correlations neighbors_threshold
correction_data_generator.compute_clusters(maximal_size=maximal_size,
                                           method=clustering_method,
                                           method_kwargs=clustering_function_arguments
                                           )
clusters = correction_data_generator.clusters_list

# Choose method_name for finding neighborhoods and its kwargs
# NOTE: see descriptions of the relevant functions for available options
# NOTE 2: this is method_name used in Ref. [0.5]
neighborhoods_method = 'pairwise'
neighborhoods_function_arguments = {'neighbors_threshold': threshold_neighbors,
                                    'show_progress_bar': True}
# compute neighborhoods based on correlations neighbors_threshold
neighborhoods = correction_data_generator.find_all_neighborhoods(maximal_size=maximal_size,
                                                                 method=neighborhoods_method,
                                                                 method_kwargs
                                                                 =neighborhoods_function_arguments)

anf.cool_print('DONE', '\n', 'green')

# PRINT OBTAINED NOISE MODEL
print('____')
anf.cool_print('Used method for clusters construction:', clustering_method, 'red')
anf.cool_print('with kwargs:', clustering_function_arguments, 'red')
anf.cool_print('Used method for neighborhoods construction:', neighborhoods_method, 'red')
anf.cool_print('with kwargs:', neighborhoods_function_arguments, 'red')
print('____')
anf.cool_print('Clusters:', clusters)
anf.cool_print('Neighborhoods:', neighborhoods)
print()

# Based on obtained noise model, get correction data for pairs of qubits
# (hence for 2-local Hamiltonians)
anf.cool_print('Constructing correction data for pairs of qubits', '...', 'green')
correction_data = correction_data_generator.get_pairs_correction_data(all_pairs,
                                                                      show_progress_bar=True)
anf.cool_print('DONE', '\n', 'green')

if saving:
    # Create dictionary to be saved
    dictionary_to_save = {'true_qubits': true_qubits, 'list_of_qubits': list_of_qubits,
                          'results_dictionary_preprocessed': dictionary_results,
                          'marginals_dictionary': correction_data_generator.marginals_dictionary,
                          'noise_matrices_dictionary':
                              correction_data_generator.noise_matrices_dictionary,
                          'clusters_labels_list': clusters, 'neighborhoods': neighborhoods,
                          'correction_data': correction_data}

    # Save results
    directory = tests_directory + 'mitigation_on_marginals/' + backend_name + \
                '/number_of_qubits_%s' % number_of_qubits + '/' + date + '/DDOT/'

    anf.save_results_pickle(dictionary_to_save,
                            directory,
                            '04_test_results_correction_data')
