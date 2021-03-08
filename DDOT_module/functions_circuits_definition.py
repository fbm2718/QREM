"""
Created on 01.03.2021

@authors: Filip B. Maciejewski, Tomek Rybotycki, Oskar Słowik
@contact: filip.b.maciejewski@gmail.com
"""

import numpy as np
import qiskit as qkt
from collections import Counter
import QREM


def get_random_bitstring(number_of_qubits):
    """Return random bitstring in the form of list of 0s and 1s
    :param number_of_qubits (int): number of qubits!

    :return: (list of str ('0' or '1') of length len(number_of_qubits))
    """
    #TODO: decide whether we want string or int. Probably STR is better because later we pass names of circuits to qiskit (and they are strings) so this avoid later data conversions.


    return [str(np.random.randint(0,2)) for i in range(number_of_qubits)]


def get_random_ddot_family(number_of_qubits, number_of_circuits, method = 'random'):
    """Return list of random bitstrings
    :param number_of_qubits (int): number of qubits!
    :param number_of_circuits (int): number of circuits in the collection. The first two circuits are always taken to be "000..." and "111..."
    :param method (str): possible values:
    'random' - uses random combinations of bitstrings
    :#TODO:  1 'hash_functions" - uses random hash functions (ADD LATER)

    #TODO: 2 The lists of strings are later concateneted to strings (e.g., ['0','0','1'] -> '001") using function "get_circuits_description_from_family".
    #TODO: 2 The initial format is better to analyze the properties of the family while the final format is better to label circuits...
    #TODO: 2 we might wish to change this



    :return: (list of (lists of str ('0' or '1') of length len(number_of_qubits)) of length number_of_circuits)
    """

    zeros, ones = [str(0) for i in range(number_of_qubits)], [str(1) for i in range(number_of_qubits)]

    if method == 'random':
        ddot_family = [zeros, ones] + [get_random_bitstring(number_of_qubits) for k in
                                                       range(number_of_circuits-2)]
    elif method == 'hash_functions':
        raise ValueError("Hash functions are to be added, use 'random' method instead")

    return ddot_family


def get_circuits_description_from_family(ddot_family):
    """Return description of DDOT circuits to be passed to "get_DDOT_circuits_qiskit"

    :param family (list of lists of str): not concatenated description of DDOT circuits returned by function get_random_ddot_family


    :return: dictionary for which each KEY is the description of circuit and VALUE is the number of times it occurs in the family

    #TODO: 1 this dictionary is needed in case there are some circuits which are implemented more than once.
    #TODO: 1 If the collection is random, then for high number of qubits this is unlikely. However, some heurisitc balancing
    #TODO: 1 methods might cause it to happen. This is important when later addresing RESULTS of experiments by name
    #TODO: 1 (if this is not done with caution, one might take the data from the same experiment twice)
    """



    circuits_description = dict(Counter([''.join(l) for l in ddot_family]))

    
    return circuits_description


def get_DDOT_circuits_qiskit(qubit_indices, 
                             quantum_register_size, 
                             circuits_description,
                             number_of_repetitions=1,
                             end_with_measurement = True):
    """Return list qiskit circuits implementing DDOT on subset of qubits with given description
    :param qubit_indices (list of ints): indices of the qubits on which DDOT should be implemented
    :param quantum_register_size (int): total number of qubits in a device (or simulator) -- has to be larger or equal max(qubit_indices)!
    :param circuits_description (dictionary): dictionary returned by function "get_circuits_description_from_family"
    :param (optional) number_of_repetitions (int): how many times each circuit should be implemented. Default is 1.
    :param (optional) end_with_measurement (Boolean): specify whether add measurement at the end of the circuit. Default is True


    :return: circuits (list of qiskit.QuantumCircuit): list of qikist circuits that implement DDOT
    :return: circuit_names (list of strings): descriptions of circuits
    """

    #How many qubits are "active" (i.e., the DDOT is performed on them)
    number_of_qubits = len(circuits_description[0])

    #Enumerate active qubits to be consistent with circuits description
    enumerated_qubits = dict(enumerate(qubit_indices))

    circuit_names = list(circuits_description.keys())


    circuits, quantum_circuit_names = [], []

    for circuit_description_index in range(len(circuit_names)):
        #Get circuit name
        circuit_string_now = circuit_names[circuit_description_index]

        #Get number of circuits repetitions which is "numbeer of occurances in the family" TIMES "number of repetitions"
        circuit_reps_now = circuits_description[circuit_string_now]*number_of_repetitions

        for rep_index in range(circuit_reps_now):
            qreg = qkt.QuantumRegister(size=quantum_register_size)
            creg = qkt.ClassicalRegister(size=len(qubit_indices))

            circuit_name_now = 'circuit_' + str(circuit_string_now) + '_no%s' % rep_index

            quantum_circuit = qkt.QuantumCircuit(qreg, creg, name=circuit_name_now)

            #Go through active qubits
            for ind, qind in enumerated_qubits.items():
                if circuit_string_now[int(ind)] in ['1',1,'x','X']:
                    quantum_circuit.x(qreg[qind])
                elif circuit_string_now[int(ind)] in ['0',0,'I','i','Id','id']:
                    pass
                else:
                    raise ValueError("Wrong circuit string: "+circuit_string_now)
    
                if end_with_measurement:
                    quantum_circuit.measure(qreg[int(qind)], creg[int(ind)])

            circuits.append(quantum_circuit)
            circuit_names.append(circuit_name_now)

    return circuits, circuit_names




def get_overlapping_QDT_single_qubit_circuits_qiskit(qubit_indices,
                                             quantum_register_size,
                                             probe_kets = [],
                                             number_of_repetitions=1,
                                             end_with_measurement = True):

    """Return list qiskit circuits implementing Quantum Detector Tomography (QDT) in parallel on all qubits
    :param qubit_indices (list of ints): indices of the qubits on which QDT should be implemented
    :param quantum_register_size (int): total number of qubits in a device (or simulator) -- has to be larger or equal max(qubit_indices)!
    :param (optional) list of vectors: list of vectors describing single-qubit pure quantum states to be used in QDT. Should be informationally-complete (i.e., span single-qubit states). Default is empty list which implements overcomplete Pauli basis (i.e., all eigenstates of Pauli matrices)
    :param (optional) number_of_repetitions (int): how many times each circuit should be implemented. Default is 1.
    :param (optional) end_with_measurement (Boolean): specify whether add measurement at the end of the circuit. Default is True


    :return: circuits (list of qiskit.QuantumCircuit): list of qikist circuits that implement parallel single-qubit QDT
    :return: circuit_names (list of strings): descriptions of circuits
    """


    circuits, circuit_names = [], []

    #How many qubits are "active" (i.e., the DDOT is performed on them)
    number_of_qubits = len(qubit_indices)

    #Enumerate active qubits to be consistent with circuits description
    enumerated_qubits = dict(enumerate(qubit_indices))

    if len(probe_kets)==0:
        probe_kets = QREM.povmtools.pauli_probe_eigenkets

    unitaries = [QREM.povmtools.get_unitary_change_ket_qubit(ket) for ket in probe_kets]


    names_strings = ['I', 'X', 'p', 'm', '1', '0']


    for unitary_ind in range(len(unitaries)):
        list_name = [names_strings[unitary_ind] for s in range(quantum_register_size)]

        for rep_index in range(number_of_repetitions):
            string_name = 'circuit_'+''.join(list_name)+'_no%s'%rep_index
            qreg = qkt.QuantumRegister(size=quantum_register_size)
            creg = qkt.ClassicalRegister(size=quantum_register_size)
            quantum_circuit = qkt.QuantumCircuit(qreg, creg, name=string_name)

            for ind, qind in enumerated_qubits.items():
                quantum_circuit, qreg = QREM.qiskit_utilities.add_gate_to_circuit(quantum_circuit,
                                                                                  qreg,
                                                                                  qind,
                                                                                  unitaries[unitary_ind])

                if end_with_measurement:
                    quantum_circuit.measure(qreg[int(qind)], creg[int(ind)])


            circuits.append(quantum_circuit)
            circuit_names.append(string_name)

    return circuits, circuit_names











