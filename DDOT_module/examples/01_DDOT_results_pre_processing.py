"""
Created on 03.05.2021

@author: Filip Maciejewski
@contact: filip.b.maciejewski@gmail.com
"""

import os, pickle
from QREM import ancillary_functions as anf
from QREM.DDOT_module.child_classes.ddot_marginal_analyzer_vanilla import DDOTMarginalsAnalyzer



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

number_of_qubits = len(true_qubits)

# qubits will be labeled from 0 to len(true_qubits)-1
list_of_qubits = range(number_of_qubits)

# in this exemplary file, the "results_dictionary_preprocessed" key is pre-processed dictionary
# with experimental results that assumes qubits are labeled from 0 to len(true_qubits)-1

# the dictionary has the structure where:
# KEY = label for experiment (in DDOT, it is bitstring denoting INPUT state)
# VALUE = counts dictionary, where each KEY is bitstring denoting measurement outcome, and VALUE is
#        number of occurrences
dictionary_results = dictionary_data['converted_dict']

# get instance of marginals analyzer for ddot experiments
marginals_analyzer_ddot = DDOTMarginalsAnalyzer(dictionary_results,
                                                reverse_counts)

# get indices of all qubit pairs (in ascending order)
all_pairs = [[i, j] for i in list_of_qubits for j in list_of_qubits if j > i]

# compute marginal distributions for all experiments and all qubit pairs
# showing progress bar requires 'tqdm' package
marginals_analyzer_ddot.compute_all_marginals(all_pairs, show_progress_bar=True)

# compute average noise matrices on all qubits pairs;
# this will be used for initial noise analysis
# showing progress bar requires 'tqdm' package
marginals_analyzer_ddot.compute_subset_noise_matrices(all_pairs, show_progress_bar=True)



# dictionary_save = {'true_qubits': true_qubits,
#                    'list_of_qubits': list_of_qubits,
#                    'results_dictionary_preprocessed': dictionary_results,
#                    'marginals_dictionary_pairs': marginals_analyzer_ddot.marginals_dictionary,
#                    'noise_matrices_dictionary_pairs': marginals_analyzer_ddot.noise_matrices_dictionary
#                    }
#
#
# date_save = '2020_05_04'
# directory = tests_directory + 'DDOT/' + backend_name + '/N%s' % number_of_qubits + '/' + date_save + '/'
#
#
# from povms_qi import povm_data_tools as pdt
# pdt.Save_Results_simple(dictionary_save,
#                         directory,
#                         'test_results')