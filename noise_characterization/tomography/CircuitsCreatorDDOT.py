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

from typing import Optional, List, Dict, Union
from collections import Counter
from functions import povmtools

__valid_SDK_names__ = ['qiskit']
__valid_experiments_names__ = ['DDOT', 'DDT', 'QDOT', 'QDT']

import numpy as np


class CircuitsCreatorDDOT:
    """
    This is general class for creating quantum circuits implementing Diagonal Detector Overlapping
    Tomography circuits (DDOT).

    NOTE: please note that this class' is used only to create QuantumCircuit objects that can be passed
    to execution on some backend. It is assumed that specific labels describing circuits (see below)
    already have been generated.

    MAIN PARAMETERS:
    :param SDK_name:
    possible values:
    1. 'qiskit'

    :param qubit_indices: one always needs to provide list of qubit indices on which experiments
                          should be performed (this does not need to be the whole device)

    :param circuits_labels: main argument to the class, this is list of symbolic circuits description

    Two formats are accepted:
    1. List[List[int]] - integer indicates gate to be implemented on
                         corresponding qubit.
                         For example: circuit_labels = [[0,1,0],
                                                        [0,0,1]]

                         will implement circuits:
                         gate "identity" on qubit_indices[0],
                         gate "X" on qubit_indices[1],
                         gate "identity" on qubit_indices[2]

                         gate "identity" on qubit_indices[0],
                         gate "identity" on qubit_indices[1],
                         gate "X" on qubit_indices[2]

    2. Dict[str,int] - dictionary where each KEY is string of integers describing
                      a circuit (in manner analogous to described above),
                      and VALUE is the number of times that this circuit should be implemented




    """

    def __init__(self,
                 SDK_name: str,
                 qubit_indices: List[int],
                 circuits_labels: Union[Dict[str, int], List[List[int]]],
                 number_of_repetitions: Optional[int] = 1,
                 quantum_register_size: Optional[int] = None,
                 classical_register_size: Optional[int] = None,
                 add_barriers=True):

        """
        :param SDK_name: see class' description
        :param qubit_indices: see class' description
        :param circuits_labels: see class' description
        :param number_of_repetitions: each circuit in circuit_labels will be implemented
                                      this number of times
                                      NOTE: if there are doubles in the list, they will be counted as
                                        multiple circuits.
                                        For example: circuit_labels = [ [0,1,0], [0,1,0] ]
                                        with number_of_repetitions = 3
                                        will implement 6 circuits [0,1,0].
        :param quantum_register_size:
        :param classical_register_size:
        :param add_barriers:
        """

        if SDK_name not in __valid_SDK_names__:
            raise ValueError('Backend: ' + SDK_name + ' is not supported yet.')

        self._SDK_name = SDK_name
        self._experiment_name = "DDOT"
        self._qubit_indices = qubit_indices

        if isinstance(circuits_labels, list):
            circuits_labels_strings_list = [''.join([str(symbol) for symbol in symbols_list]) for
                                            symbols_list
                                            in
                                            circuits_labels]
            circuits_labels = dict(zip(Counter(circuits_labels_strings_list).keys(),
                                       Counter(circuits_labels_strings_list).values()))
            if number_of_repetitions > 1:
                for key in circuits_labels.keys():
                    circuits_labels[key] *= number_of_repetitions

        self._circuit_labels = circuits_labels
        self._quantum_register_size = quantum_register_size
        self._classical_register_size = classical_register_size
        self._add_barriers = add_barriers


    @property
    def circuits_labels_dictionary(self)->Dict[str,int]:
        return self._circuit_labels

    @staticmethod
    def _add_measurements_qiskit(
            circuit_object,
            qreg,
            creg,
            qubit_indices):
        for qubit_index in range(len(qubit_indices)):
            circuit_object.measure(qreg[qubit_indices[qubit_index]], creg[qubit_index])

        return circuit_object


    def get_circuits(self,
                     add_measurements: Optional[bool] = True):
        """

        Returns quantum circuits as a list.

        Circuits are later identified by names for which we use the following convention:

        circuit_name = "experiment name" + "-" + "circuit label"+
        "no"+ "integer identifier for multiple implementations of the same circuit"

        for example the circuit can have name:
        "DDOT-010-no3"

        which means that this experiment is Diagonal Detector Overlapping Tomography (DDOT),
        the circuit implements state "010" (i.e., gates iden, X, iden on qubits 0,1,2), and
        in the whole circuits sets this is the 4th (we start counting from 0) circuit that implements
        that particular state.

        :param add_measurements:
        :return:
        """

        if self._SDK_name == 'qiskit':
            from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
        else:
            raise ValueError('SDK :' + self._SDK_name + " not supported yet.")


        qubit_indices = self._qubit_indices

        quantum_register_size, classical_register_size = self._quantum_register_size, \
                                                         self._classical_register_size

        circuits_labels = self._circuit_labels

        qubit_indices = sorted(
            qubit_indices)  # Sort to ensure, that results can easily be interpreted.

        if quantum_register_size is None:
            quantum_register_size = max(qubit_indices) + 1

        if classical_register_size is None:
            classical_register_size = len(qubit_indices)

        all_circuits = []

        unique_circuit_labels = list(circuits_labels.keys())

        # outer loop is for copies of DDOT experiment
        # for repetition_index in range(number_of_repetitions):
        # inner loop goes through all circuits in QDT experiments and prepares them
        for circuit_label_string in unique_circuit_labels:
            # this loop goes over multiple instances of the same experiments (if they exist)
            circuit_label_list = list(circuit_label_string)
            # print(circuit_label_list)
            # raise KeyError
            for multiple_circuits_counter in range(circuits_labels[circuit_label_string]):
                if self._SDK_name in ['qiskit', 'Qiskit']:
                    # create registers
                    qreg, creg = QuantumRegister(quantum_register_size), \
                                 ClassicalRegister(classical_register_size)

                    # create quantum circuit with nice name
                    circuit_name = self._experiment_name + "-" \
                                   + circuit_label_string + 'no%s' % multiple_circuits_counter

                    circuit_object = QuantumCircuit(qreg,
                                                    creg,
                                                    name=circuit_name)

                    for qubit_index in range(len(circuit_label_list)):
                        qubit_now = qubit_indices[qubit_index]
                        label_now = circuit_label_list[qubit_now]

                        if label_now == '0' or label_now == 0:
                            circuit_object.id(qubit_now)
                        elif label_now == '1' or label_now == 1:
                            circuit_object.x(qubit_now)
                        else:
                            raise ValueError('Wrong circuit label: ', circuit_label_list)

                    # get barrier to prevent compiler from making changes
                    if self._add_barriers:
                        circuit_object.barrier()

                    if add_measurements:
                        circuit_object = self._add_measurements_qiskit(circuit_object,
                                                                       qreg,
                                                                       creg,
                                                                       qubit_indices)

                    all_circuits.append(circuit_object)
        return all_circuits
