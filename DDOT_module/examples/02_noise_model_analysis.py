"""
Created on 04.05.2021

@author: Filip Maciejewski
@contact: filip.b.maciejewski@gmail.com
"""

import os, pickle
from QREM import ancillary_functions as anf
# from QREM.DDOT_module.child_classes.ddot_marginal_analyzer_vanilla import DDOTMarginalsAnalyzer
from QREM.DDOT_module.child_classes.noise_model_generator_vanilla import NoiseModelGenerator

module_directory = anf.get_module_directory()
tests_directory = module_directory + '/data_for_tests/'

# data used for testing
backend_name = 'ASPEN-8'
number_of_qubits = 23
date = '2020_05_04'

# specify whether count names are read from right to left (convention used by IBM)
reverse_counts = False

directory = tests_directory + 'DDOT/' + backend_name + '/N%s' % number_of_qubits + '/' + date + '/'

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

# initialize noise model generator
noise_model_analyzer = NoiseModelGenerator(results_dictionary_ddot=dictionary_results,
                                           reverse_counts=reverse_counts,
                                           number_of_qubits=number_of_qubits,
                                           marginals_dictionary=marginal_dictionaries_pairs,
                                           noise_matrices_dictionary=noise_matrices_dictionary_pairs)

# compute correlations table for qubits pairs
noise_model_analyzer.compute_correlations_table_pairs()
"""
Correlations are defined as:

c_{j -> i} = 1/2 * || \Lambda_{i}^{Y_j = '0'} - \Lambda_{i}^{Y_j = '0'}||_{l1}

Where \Lambda_{i}^{Y_j} is an effective noise matrix on qubit "i" (averaged over all other of
qubits except "j"), provided that input state of qubit "j" was "Y_j". Hence, c_{j -> i}
measures how much noise on qubit "i" depends on the input state of qubit "j".
"""

correlations_table_pairs = noise_model_analyzer.correlations_table_pairs
anf.print_array_nicely(correlations_table_pairs[0:10, 0:10], 3)

# #
threshold_clusters = 0.06
threshold_neighbors = 0.02
maximal_size = 5
#
# #compute clusters using naive method based on correlations treshold
noise_model_analyzer.compute_clusters_naive(max_size=maximal_size,
                                            threshold=threshold_clusters
                                            )

clusters_naive = noise_model_analyzer.clusters_list

neighborhoods_naive = noise_model_analyzer.find_all_neighborhoods_naive(maximal_size=maximal_size,
                                                                        threshold=threshold_neighbors)
neighborhoods_naive_smarter = noise_model_analyzer.find_all_neighborhoods(maximal_size=maximal_size,
                                                                          chopping_threshold=0.02,
                                                                          show_progress_bar=True)

print(neighborhoods_naive_smarter)

noise_model_analyzer.compute_clusters_heuristic(max_size=3,
                                                version='v1')

clusters_list_heuristic = noise_model_analyzer.clusters_list

noise_model_analyzer.find_all_neighborhoods(maximal_size=maximal_size,
                                            chopping_threshold=0.02,
                                            show_progress_bar=True)

neighborhoods_heuristic = noise_model_analyzer.neighborhoods

no_clusters = [[qi] for qi in list_of_qubits]

noise_model_analyzer.clusters_list = no_clusters

noise_model_analyzer.find_all_neighborhoods_naive(maximal_size=maximal_size,
                                                  threshold=threshold_neighbors)

neighborhoods_no_clusters_naive = noise_model_analyzer.neighborhoods

noise_model_analyzer.find_all_neighborhoods(maximal_size=maximal_size,
                                            chopping_threshold=0.02,
                                            show_progress_bar=True)

neighborhoods_no_clusters_smarter = noise_model_analyzer.neighborhoods

from povms_qi.ancillary_functions import cool_print

cool_print('Clusters naive:', clusters_naive)
cool_print('Clusters heuristic:', clusters_list_heuristic)
print()
cool_print('Neighborhoods naive:', neighborhoods_naive)
cool_print('Neighborhoods naive smarter:', neighborhoods_naive_smarter)
cool_print('Neighborhoods heuristic:', neighborhoods_heuristic)
cool_print('Neighborhoods naive no clusters:', neighborhoods_no_clusters_naive)
cool_print('Neighborhoods smarter no clusters:', neighborhoods_no_clusters_smarter)
print()

# dictionary_to_save = {}
dictionary_to_save = {'true_qubits': true_qubits,
                      'list_of_qubits': list_of_qubits,
                      'results_dictionary_preprocessed': dictionary_results,
                      'marginals_dictionary_pairs': noise_model_analyzer.marginals_dictionary,
                      'noise_matrices_dictionary_pairs': noise_model_analyzer.noise_matrices_dictionary,
                      'clusters_list_naive': clusters_naive,
                      'neighborhoods_naive': neighborhoods_naive,
                      'neighborhoods_naive_smarter': neighborhoods_naive_smarter,
                      'clusters_list_heuristic': clusters_list_heuristic,
                      'neighborhoods_heuristic': neighborhoods_heuristic,
                      'no_clusters': no_clusters,
                      'neighborhoods_no_clusters_naive': neighborhoods_no_clusters_naive,
                      'neighborhoods_no_clusters_smarter': neighborhoods_no_clusters_smarter
                      }
# diction
#
date_save = '2020_05_04'
directory = tests_directory + 'DDOT/' + backend_name + '/N%s' % number_of_qubits + '/' + date_save + '/'

from povms_qi import povm_data_tools as pdt

pdt.Save_Results_simple(dictionary_to_save,
                        directory,
                        'test_results')
