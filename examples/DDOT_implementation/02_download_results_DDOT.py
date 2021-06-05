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

from functions.functions_SDKs.qiskit import qiskit_utilities as qkt_utils
from functions import functions_data_analysis as fdt, ancillary_functions as anf

"""
This script is solely for downloading and processing results of DDOT experiments performed on 
hardware backend of qiskit. 
"""

date_string = "2021-06-03 14_09_53"
experiment_name = 'DDOT'
backend_name = 'ibmq_16_melbourne'

number_of_qubits = 15
circuits_amount = 375
jobs_directory = anf.get_module_directory() + "/saved_data/tomography_results/DDOT/" \
                 + backend_name + "/number_of_qubits_%s" % number_of_qubits + "/job_IDs/" \
                 + date_string + "/"

files = sorted(os.listdir(jobs_directory))

job_IDs_list = []

for job_file in files:
    with open(jobs_directory + job_file, 'rb') as filein:
        dictionary_data = pickle.load(filein)
        job_IDs_list.append(dictionary_data['job_ID'])

circuits_amount = dictionary_data['number_of_circuits']

jobs_downloaded = qkt_utils.download_multiple_jobs(backend_name,
                                                   job_IDs_list)

unprocessed_results = qkt_utils.get_counts_from_jobs(jobs_downloaded)

save_directory_raw_counts = anf.get_module_directory() + "/saved_data/tomography_results/DDOT/" \
                            + backend_name + "/number_of_qubits_%s" % number_of_qubits + "/counts_raw/" \
                            + date_string + "/"

file_name = 'circuits_amount%s' % circuits_amount + '_' + date_string

dictionary_to_save = {'job_IDs': job_IDs_list,
                      'true_qubits': dictionary_data['true_qubits'],
                      'experiment_name': "DDOT",
                      'circuits_labels': dictionary_data['circuits_labels'],
                      'SDK_name': backend_name,
                      'date': date_string,
                      'unprocessed_counts': unprocessed_results}

anf.save_results_pickle(dictionary_to_save=dictionary_to_save,
                        directory=save_directory_raw_counts,
                        custom_name=file_name)

processed_results = fdt.convert_counts_overlapping_tomography(counts_dictionary=unprocessed_results,
                                                              experiment_name=experiment_name)

save_directory_counts_processed = anf.get_module_directory() + "/saved_data/tomography_results/DDOT/" \
                                  + backend_name + "/number_of_qubits_%s" % number_of_qubits \
                                  + "/counts_processed/" + date_string + "/"

dictionary_to_save['results_dictionary_preprocessed'] = processed_results

anf.save_results_pickle(dictionary_to_save=dictionary_to_save,
                        directory=save_directory_counts_processed,
                        custom_name=file_name)
