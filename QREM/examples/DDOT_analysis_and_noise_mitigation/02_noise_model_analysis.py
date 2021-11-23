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
import os, pickle
from QREM.functions import ancillary_functions as anf
from QREM.noise_characterization.modeling.NoiseModelGenerator import NoiseModelGenerator
from QREM.functions.ancillary_functions import cool_print

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

directory = tests_directory + 'mitigation_on_marginals/' + backend_name + '/number_of_qubits_%s' % number_of_qubits + '/' + date + '/DDOT/'

files = os.listdir(directory)
with open(directory + '02_test_results_noise_matrices_pairs.pkl', 'rb') as filein:
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
marginal_dictionaries_pairs = dictionary_data['marginals_dictionary_pairs']

# dictionary for which each KEY is label for qubits subset
# and VALUE is effective noise matrix on that subset
# in this example, we precomputed noise matrices for all pairs of qubits
noise_matrices_dictionary_pairs = dictionary_data['noise_matrices_dictionary_pairs']

# initialize noise model generator
noise_model_analyzer = NoiseModelGenerator(results_dictionary_ddot=dictionary_results,
                                           bitstrings_right_to_left=bitstrings_right_to_left,
                                           number_of_qubits=number_of_qubits,
                                           marginals_dictionary=marginal_dictionaries_pairs,
                                           noise_matrices_dictionary=noise_matrices_dictionary_pairs)

# compute correlations table for qubits pairs
noise_model_analyzer.compute_correlations_table_pairs()
"""
Correlations are defined as:

c_{j -> i_index} = 1/2 * || \Lambda_{i_index}^{Y_j = '0'} - \Lambda_{i_index}^{Y_j = '0'}||_{l1}

Where \Lambda_{i_index}^{Y_j} is an effective noise matrix on qubit "i_index" (averaged over all other of
qubits except "j"), provided that input state of qubit "j" was "Y_j". Hence, c_{j -> i_index}
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

# set maximal size of cluster+neighborhood set
maximal_size = 5

# Choose clustering method_name and its kwargs
# NOTE: see descriptions of the relevant functions for available options
# NOTE 2: this is method_name used in Ref. [0.5]
clustering_method = 'pairwise'
clustering_function_arguments = {'cluster_threshold': threshold_clusters}

anf.cool_print('Got data, computing clusters...','...')
# compute clusters based on correlations treshold
noise_model_analyzer.compute_clusters(maximal_size=maximal_size,
                                      method=clustering_method,
                                      method_kwargs=clustering_function_arguments
                                      )

clusters = noise_model_analyzer.clusters_list

# Choose method_name for finding neighborhoods and its kwargs
# NOTE: see descriptions of the relevant functions for available options
# NOTE 2: this is method_name used in Ref. [0.5]
neighborhoods_method = 'pairwise'
neighborhoods_function_arguments = {'neighbors_threshold': threshold_neighbors,
                                    'show_progress_bar': True}
# compute neighborhoods based on correlations treshold
neighborhoods = noise_model_analyzer.find_all_neighborhoods(maximal_size=maximal_size,
                                                            method=neighborhoods_method,
                                                            method_kwargs=
                                                            neighborhoods_function_arguments)

cool_print('Clusters:', clusters)
cool_print('Neighborhoods:', neighborhoods)
print()
if saving:
    dictionary_to_save = {'true_qubits': true_qubits,
                          'list_of_qubits': list_of_qubits,
                          'results_dictionary_preprocessed': dictionary_results,
                          'marginals_dictionary': noise_model_analyzer.marginals_dictionary,
                          'noise_matrices_dictionary': noise_model_analyzer.noise_matrices_dictionary,
                          'clusters_labels_list': clusters,
                          'neighborhoods': neighborhoods
                          }

    directory = tests_directory + 'mitigation_on_marginals/' + backend_name + '/number_of_qubits_%s' % number_of_qubits + '/' + date + '/DDOT'

    anf.save_results_pickle(dictionary_to_save,
                            directory,
                            '03_test_results_noise_models')
