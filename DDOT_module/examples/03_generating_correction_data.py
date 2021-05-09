"""
Created on 04.05.2021

@author: Filip Maciejewski
@contact: filip.b.maciejewski@gmail.com
"""

import os, pickle
from QREM import ancillary_functions as anf
from QREM.DDOT_module.child_classes.correction_data_generator import CorrectionDataGenerator

# from QREM.DDOT_module.child_classes.marginals_corrector import MarginalsCorrector

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

# NAIVE CLUSTERS
clusters_list_naive, neighborhoods_naive = dictionary_data['clusters_list_naive'], dictionary_data[
    'neighborhoods_naive']

correction_data_generator_naive = CorrectionDataGenerator(results_dictionary_ddot=dictionary_results,
                                                          reverse_counts=reverse_counts,
                                                          number_of_qubits=number_of_qubits,
                                                          marginals_dictionary=marginal_dictionaries_pairs,
                                                          clusters_list=clusters_list_naive,
                                                          neighborhoods=neighborhoods_naive,
                                                          noise_matrices_dictionary=
                                                          noise_matrices_dictionary_pairs)

all_pairs = [[qi, qj] for qi in list_of_qubits for qj in list_of_qubits if qj > qi]

correction_data_naive = correction_data_generator_naive.get_pairs_correction_data(all_pairs,
                                                                                  show_progress_bar=True)

# NO CLUSTERS
no_clusters, neighborhoods_no_clusters = dictionary_data['no_clusters'], \
                                         dictionary_data[
                                             'neighborhoods_no_clusters_naive']

correction_data_generator_no_clusters = CorrectionDataGenerator(
    results_dictionary_ddot=dictionary_results,
    reverse_counts=reverse_counts,
    number_of_qubits=number_of_qubits,
    marginals_dictionary=marginal_dictionaries_pairs,
    clusters_list=no_clusters,
    neighborhoods=neighborhoods_no_clusters,
    noise_matrices_dictionary=
    noise_matrices_dictionary_pairs)

correction_data_no_clusters = correction_data_generator_no_clusters.get_pairs_correction_data(
    all_pairs,
    show_progress_bar=True)

dictionary_data['correction_data_naive'] = correction_data_naive
dictionary_data['correction_data_no_clusters'] = correction_data_no_clusters

# marginals_corrector
date_save = '2020_05_04'
directory = tests_directory + 'DDOT/' + backend_name + '/N%s' % number_of_qubits + '/' + date_save + '/DDOT/'

from povms_qi import povm_data_tools as pdt

pdt.Save_Results_simple(dictionary_data,
                        directory,
                        'test_results')
