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

import numpy as np
from QREM.noise_characterization.tomography.CircuitsCreatorDDOT import CircuitsCreatorDDOT
from functions import functions_data_analysis as fdt, ancillary_functions as anf
from functions.functions_SDKs.qiskit import qiskit_utilities as qkt_utils


"""
This script creates collection of qiskit's QuantumCircuit objects that implement 
Diagonal Detector Overlapping Tomography (DDOT) (see [0.5]). 

GENERAL WORKFLOW:

1. Load circuits' labels created previously (see 00_creating_circuits_DDOT).
2. Get instance of CircuitsCreatorDDOT and run function that creates circuits.
3. Divide circuits into batches and send job for execution on chosen backend.
"""



# backend name
backend_name = 'qasm_simulator'

SDK_name = 'qiskit'

# Define number of qubits you wish to create DDOT circuits for
number_of_qubits = 5
qubit_indices = list(range(number_of_qubits))

# Locality of subsets we wish to investigate. For example, k=2 will implement all computational-basis
# states (00, 01, 10, 11) on ALL qubit pairs.
subsets_locality = 2

directory = anf.get_module_directory() + '/saved_data/data_circuits_collections/DDOT/' + \
            '/locality_%s' % subsets_locality + '/number_of_qubits_%s' % number_of_qubits + '/'

files = sorted(os.listdir(directory))

anf.cool_print('Available files:\n', files)
anf.cool_print('Choosing file:\n',files[-1])
with open(directory + files[-1], 'rb') as filein:
    dictionary_data = pickle.load(filein)

circuits_labels = dictionary_data['circuits_list']

circuits_amount = len(circuits_labels)

circuits_creator = CircuitsCreatorDDOT(SDK_name=SDK_name,
                                       qubit_indices=qubit_indices,
                                       circuits_labels=circuits_labels,
                                       number_of_repetitions=1,
                                       add_barriers=True
                                       )

# print(circuits_labels)
DDOT_circuits_list = circuits_creator.get_circuits(add_measurements=True)

DDOT_circuits_dictionary = circuits_creator.circuits_labels_dictionary

circuits_per_job = 75

jobs_amount = int(np.ceil(len(DDOT_circuits_list) / circuits_per_job))

batches = []

counter = 0
for batch_index in range(jobs_amount):
    circuits_now = DDOT_circuits_list[counter * circuits_per_job:(counter + 1) * circuits_per_job]
    batches.append(circuits_now)
    counter += 1

date_string = anf.gate_proper_date_string()
file_name = 'circuits_amount%s' % len(DDOT_circuits_list) + '_' + date_string

save_directory = anf.get_module_directory() + "/saved_data/tomography_results/DDOT/" \
                 + backend_name + "/number_of_qubits_%s" % number_of_qubits + "/job_IDs/" \
                 + date_string + "/"

dictionary_to_save = {'circuits_labels': circuits_labels,
                      'true_qubits': qubit_indices,
                      'number_of_circuits': circuits_amount}

saving_IDs_dictionary = {'saving': True,
                         'directory': save_directory,
                         'file_name': file_name,
                         'dictionary_to_save': {'circuits_labels': circuits_labels,
                                                'true_qubits': qubit_indices,
                                                'number_of_circuits': circuits_amount}}

anf.cool_print('Experiment name:', "Diagonal Detector Overlapping Tomography (DDOT)")
anf.cool_print('Locality: ', subsets_locality)
anf.cool_print('Backend:', backend_name, 'red')
anf.cool_print('Number of circuits:', len(DDOT_circuits_list))
anf.cool_print('Number of batches:', len(batches))

if anf.query_yes_no("Do you want to run?"):
    pass
else:
    raise KeyboardInterrupt("OK")

jobs = qkt_utils.run_batches(batches=batches,
                             backend_name=backend_name,
                             shots=8192,
                             saving_IDs_dictionary=saving_IDs_dictionary)

"""
If backend is actual hardware, go to "02_download_results_DDOT.py" to download results after circuits 
have been implemented.

If backend is a simulator, we may processed results here, directly after simulation is finished:
"""

if backend_name in ['qasm_simulator', 'statevector_simulator', 'unitary_simulator']:
    dictionary_data = dictionary_to_save

    unprocessed_results = qkt_utils.get_counts_from_jobs(jobs)

    save_directory_raw_counts = anf.get_module_directory() + "/saved_data/tomography_results/DDOT/" \
                                + backend_name + "/number_of_qubits_%s" % number_of_qubits + "/counts_raw/" \
                                + date_string + "/"

    file_name = 'circuits_amount%s' % circuits_amount + '_' + date_string

    dictionary_to_save = {
        'true_qubits': dictionary_data['true_qubits'],
        'experiment_name': "DDOT",
        'circuits_labels': dictionary_data['circuits_labels'],
        'SDK_name': backend_name,
        'date': date_string,
        'unprocessed_counts': unprocessed_results}

    anf.save_results_pickle(dictionary_to_save=dictionary_to_save,
                            directory=save_directory_raw_counts,
                            custom_name=file_name)

    processed_results = fdt.convert_counts_overlapping_tomography(
        counts_dictionary=unprocessed_results,
        experiment_name="DDOT")

    save_directory_counts_processed = anf.get_module_directory() + "/saved_data/tomography_results/DDOT/" \
                                      + backend_name + "/number_of_qubits_%s" % number_of_qubits \
                                      + "/counts_processed/" + date_string + "/"

    dictionary_to_save['results_dictionary_preprocessed'] = processed_results

    anf.save_results_pickle(dictionary_to_save=dictionary_to_save,
                            directory=save_directory_counts_processed,
                            custom_name=file_name)
