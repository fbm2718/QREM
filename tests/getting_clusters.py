"""
Created on 28.04.2021

@author: fbm
@contact: filip.b.maciejewski@gmail.com
"""
import os
import pickle
from matplotlib import rc

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)

'''
Our library
'''

from DDOT_module.functions import depreciated_functions as fnma

backend_name = 'ASPEN-8'
number_of_qubits = 23
date = '2020_04_08'

directory = os.getcwd() + '/saved_data/DDOT/' + backend_name + '/N%s' % number_of_qubits + '/' + date + '/'
files = os.listdir(directory)
with open(directory + files[-1], 'rb') as filein:
    data_object = pickle.load(filein)

# Keys in dictionary:
# 'converted_dict' : dictionary for which each KEY is classical INPUT state, and VALUE is dictionary of counts (results);
#                    it is in potentially_stochastic_matrix form such that qubits are labeled from 0 to len(outcome)
# 'marginal_dicionaries_initial': dictionary for which each KEY is classical INPUT state, and VALUE is dictionary of
#                                 marginal distributions on all pairs of qubits
# 'pairs_noise_matrices_initial': dictionary for which each KEY is index of qubits pair, e.g., 'q2q13' and VALUE is
#                                 noise matrix on that pair (averaged over all other qubits)
# 'list_of_qubits': true indices of qubits in potentially_stochastic_matrix device

dictionary_data = data_object
dictionary_results = dictionary_data['converted_dict']

marginal_dictionaries_initial = dictionary_data[
    'marginal_dictionaries_initial']

pairs_noise_matrices_initial = dictionary_data['pairs_noise_matrices_initial']
true_qubits = dictionary_data['list_of_qubits']

mapped_qubits = range(len(true_qubits))

correlations_table_initial = fnma.calculate_correlations_pairs(pairs_noise_matrices_initial,
                                                               mapped_qubits)

for k in mapped_qubits:
    for l in mapped_qubits:
        cij = correlations_table_initial[k, l]
        #
        if cij < 0.02:
            correlations_table_initial[k, l] = 0

initial_clusters = fnma.get_initial_clusters(mapped_qubits,
                                             correlations_table_initial,
                                             0.06)

clusters_dict_v1 = {'q%s' % q: [] for q in range(number_of_qubits)}

for cluster in initial_clusters:
    for qi in cluster:
        for qj in cluster:
            if qi != qj:
                clusters_dict_v1['q%s' % qi].append(qj)

print(clusters_dict_v1)

# clusters_dict, neighborhoods_clusters, proper_clusters, neighborhoods_qubits = fnma.

neighborhood_treshold = 0.02
cluster_treshold = 10**6
clusters_dict, neighborhoods_clusters, proper_clusters, neighborhoods_qubits = \
    fnma.get_clusters_and_neighbors_from_pairs(
    mapped_qubits,
    correlations_table_initial,
    neighborhood_treshold=neighborhood_treshold,
    cluster_treshold=cluster_treshold)

clusters_cut = fnma.cut_subset_sizes(neighborhoods_clusters,
                                     correlations_table_initial,
                                     target_size=5)

# print(clusters_dict)
# print(neighborhoods_clusters)
# print(clusters_cut)


