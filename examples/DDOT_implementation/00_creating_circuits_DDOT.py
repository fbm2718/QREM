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

from QREM.noise_characterization.tomography.LabelsCreatorDDOT import LabelsCreatorDDOT
from functions import ancillary_functions as anf

"""
This script creates collection of symbolic circuits representation that can be used to implement 
Diagonal Detector Overlapping Tomography (DDOT) (see [0.5]). 

GENERAL WORKFLOW:

1. Specify number of qubits and DDOT locality. 
2. Get instance of LabelsCreatorDDOT and run function that creates random circuits labels.
3. Those labels will later be used to create QuantumCircuit object for implementation 
on physical hardware (see 02_running_circuits_DDOT_qiskit).

"""


# Define number of qubits you wish to create DDOT circuits for
number_of_qubits = 15

# Locality of subsets we wish to investigate. For example, k=2 will implement all computational-basis
# states (00, 01, 10, 11) on ALL qubit pairs.
subsets_locality = 5
#
# NOTE: the needed number of subsets scales like number_of_qubits^subsets_locality, so computing time
#      can get big for high locality. Moreover, sampling complexity grows exponentially with subsets
#      locality, so it is not advisable to go to subsets_locality>5.

# Specify maximal allowed number of circuits
maximal_circuits_amount = 375

# Specify whether you wish to save generated circuits
saving = True

# Initialize class that will create labels for our circuits
base_OT = LabelsCreatorDDOT(number_of_qubits=number_of_qubits,
                            subsets_locality=subsets_locality,
                            maximal_circuits_amount=maximal_circuits_amount,
                            show_progress_bars=False
                            )

# Method for finding collection of DDOT circuits.
# This will create random perfect collections
# and choose the one that minimizes the heuristic cost function
# See class' description for details
method_name = 'bruteforce_randomized'

# Number of created perfect collections (among which the best one will be chosen)
number_of_iterations = 100

# Number of circuits one wishes to add to a perfect collection in order to get better
# properties of circuits (but at the cost of adding additional circuits)
# if number of circuits in perfect collection + additional_circuits exceeds maximal_circuits_amount
# it is cut to meet the treshold.
additional_circuits = 0

# See class' description for details
method_kwargs = {'number_of_iterations': number_of_iterations,
                 'circuits_in_batch': 1,
                 'print_updates': True,
                 'optimized_quantity': {
                     'absent': 100.,
                     'minimal_amount': 1.0,
                     'spread': 0.01,
                     'SD': 0.0001,
                     'amount': 10 ** (-9)},
                 'additional_circuits': additional_circuits}

# Compute perfect collection of DDOT circuits, meaning that we want circuits that implement every
# computational basis state on each k-qubit subset at least once.
base_OT.compute_perfect_collection(method_name=method_name,
                                   method_kwargs=method_kwargs)

# get list of circuits
DDOT_circuits = base_OT.circuits_list

# Calculate various properties of circuits (and print them)
base_OT.calculate_properties_of_circuits()

circuits_properties = base_OT.circuits_properties_dictionary
anf.cool_print("Properties of the family:\n", circuits_properties)
# print(circuits_properties)


if saving and anf.query_yes_no('Do you still wish to save?'):
    dictionary_to_save = {'circuits_list': DDOT_circuits,
                          'circuits_properties': circuits_properties}

    directory = anf.get_module_directory() + '/saved_data/data_circuits_collections/DDOT/' + \
                'locality_%s' % subsets_locality + '/number_of_qubits_%s' % number_of_qubits + '/'

    file_name = 'circuits_amount%s' % len(DDOT_circuits) + '_' + anf.gate_proper_date_string()

    directory_to_save = anf.save_results_pickle(dictionary_to_save=dictionary_to_save,
                                                directory=directory,
                                                custom_name=file_name
                                                )
