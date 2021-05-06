"""
Created on 03.05.2021

@author: Filip Maciejewski
@contact: filip.b.maciejewski@gmail.com
"""

import os, pickle
from QREM import ancillary_functions as anf
from QREM.DDOT_module.child_classes.ddot_marginal_analyzer_vanilla import DDTMarginalsAnalyzer
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

# import pandas as pd
# dictionary_data = pd.read_pickle(directory+files[-1])

# raise KeyError

# in this exemplary file, the "true qubits" are labels of physical qubits
true_qubits = dictionary_data['true_qubits']
# print(dictionary_data.keys())
# true_qubits = dictionary_data['list_of_qubits']


# qubits will be labeled from 0 to len(true_qubits)-1
list_of_qubits = dictionary_data['list_of_qubits']

number_of_qubits = len(list_of_qubits)

# in this exemplary file, the "results_dictionary_preprocessed" key is pre-processed dictionary
# with experimental results that assumes qubits are labeled from 0 to len(true_qubits)-1

# the dictionary has the structure where:
# KEY = label for experiment (in DDOT, it is bitstring denoting INPUT state)
# VALUE = counts dictionary, where each KEY is bitstring denoting measurement outcome, and VALUE is
#        number of occurrences
dictionary_results = dictionary_data['results_dictionary_preprocessed']
# dictionary_results = dictionary_data['converted_dict']

# print(dictionary_data['marginals_dictionary_pairs'].keys())
# get instance of marginals analyzer for ddot experiments


# marginals_analyzer_ddot = DDOTMarginalsAnalyzer(dictionary_results,
#                                                 bitstrings_right_to_left)
#
marginals_analyzer_ddot = DDTMarginalsAnalyzer(dictionary_results,
                                               reverse_counts,
                                               marginals_dictionary=dictionary_data[
                                                    'marginals_dictionary_pairs'])

# dict = marginals_analyzer_ddot.marginals_dictionary
# print(marginals_analyzer_ddot.marginals_dictionary)
# raise KeyError


# get indices of all qubit pairs (in ascending order)
all_pairs = [[i, j] for i in list_of_qubits for j in list_of_qubits if j > i]

# compute marginal distributions for all experiments and all qubit pairs
# showing progress bar requires 'tqdm' package
# marginals_analyzer_ddot.compute_all_marginals(all_pairs, show_progress_bar=True)

# compute average noise matrices on all qubits pairs;
# this will be used for initial noise analysis
# showing progress bar requires 'tqdm' package
marginals_analyzer_ddot.compute_subset_noise_matrices_averaged(all_pairs,
                                                               show_progress_bar=True)

print(marginals_analyzer_ddot.noise_matrices_dictionary['q0q1'])

# dictionary_save = {'true_qubits': true_qubits,
#                    'list_of_qubits': list_of_qubits,
#                    'results_dictionary_preprocessed': dictionary_results,
#                    'marginals_dictionary_pairs': marginals_analyzer_ddot.marginals_dictionary,
#                    'noise_matrices_dictionary_pairs': marginals_analyzer_ddot.noise_matrices_dictionary
#                    }
# # diction
# #
# date_save = '2020_05_04'
# directory = tests_directory + 'DDOT/' + backend_name + '/N%s' % number_of_qubits + '/' + date_save + '/'
# #
# #
# from povms_qi import povm_data_tools as pdt
# pdt.Save_Results_simple(dictionary_save,
#                         directory,
#                         'test_results')


# raise KeyError


# dictionary for which each KEY is classical INPUT state, and VALUE is dictionary of
# marginal distributions on all pairs of qubits
# in this example, we precomputed marginals for all experiments and all pairs of qubits
marginal_dictionaries_pairs = dictionary_data['marginals_dictionary_pairs']

# dictionary for which each KEY is label for qubits subset
# and VALUE is effective noise matrix on that subset
# in this example, we precomputed noise matrices for all pairs of qubits
noise_matrices_dictionary_pairs = dictionary_data['noise_matrices_dictionary_pairs']

# print(marginal_dictionaries_pairs['q0q1'])
# noise_matrices_dictionary_pairs = marginals_analyzer_ddot.noise_matrices_dictionary
# print(noise_matrices_dictionary_pairs['q0q1'])

# print(noise_matrices_dictionary_pairs['q0q1'])
# raise KeyError
# initialize noise model generator
noise_model_analyzer = NoiseModelGenerator(results_dictionary_ddot=dictionary_results,
                                           bitstrings_right_to_left=reverse_counts,
                                           number_of_qubits=number_of_qubits,
                                           marginals_dictionary=marginal_dictionaries_pairs,
                                           noise_matrices_dictionary=noise_matrices_dictionary_pairs)

# print(noise_model_analyzer.noise_matrices_dictionary['q0q1'])

noise_model_analyzer.compute_correlations_table_pairs()

correlations_pair_table = noise_model_analyzer.correlations_table_pairs

# anf.print_array_nicely(correlations_pair_table[0:10, 0:10], 3)

# print(type(correlations_pair_table), correlations_pair_table.shape)
# print(type(list_of_qubits), len(list_of_qubits))
threshold = 0.06
maximal_size = 5

from QREM.DDOT_module.functions import depreciated_functions as dpf

old_clusters = dpf.get_initial_clusters(list(list_of_qubits),
                                        correlations_pair_table,
                                        threshold)

print(old_clusters)

noise_model_analyzer.compute_clusters_naive(max_size=maximal_size,
                                            threshold=threshold
                                            )


clusters_naive = noise_model_analyzer.clusters_list

neighborhoods_naive = noise_model_analyzer.find_all_neighborhoods_naive(maximal_size=maximal_size,
                                                                        threshold=0.02)
print(clusters_naive)
print(neighborhoods_naive)


neighborhoods_smart = noise_model_analyzer.find_all_neighborhoods(maximal_size=maximal_size,
                                                                  chopping_threshold=0.02,
                                                                  show_progress_bar=True)

print(neighborhoods_smart)







