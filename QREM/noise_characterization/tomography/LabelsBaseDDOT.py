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
import numpy as np

from functions import ancillary_functions as anf


class LabelsBaseDDOT:
    def __init__(self,
                 number_of_qubits: int,
                 subsets: Union[Dict[str, List[int]], List[List[int]]],
                 maximal_circuits_amount: Optional[int] = 1500,
                 show_progress_bars: Optional[bool] = True,
                 ):
        self._number_of_qubits = number_of_qubits
        self._subsets = subsets
        self._maximal_circuits_amount = maximal_circuits_amount
        self._circuits_list = []

        self._show_progress_bars = show_progress_bars

    @property
    def subsets(self) -> Union[Dict[str, List[int]], List[List[int]]]:
        return self._subsets

    @subsets.setter
    def subsets(self, subsets: Union[Dict[str, List[int]], List[List[int]]]) -> None:
        self._subsets = subsets

    @property
    def circuits_list(self) -> List[List[int]]:
        return self._circuits_list

    @circuits_list.setter
    def circuits_list(self, circuits_list: List[List[int]]) -> None:
        self._circuits_list = circuits_list

    def get_random_circuit(self):
        return [np.random.randint(0, 2) for _ in
                range(self._number_of_qubits)]

    def get_random_circuit_with_fixed_state_of_some_qubits(self,
                                                           fixed_states: Dict[int, int]):
        """
        :param fixed_states: dictionary where each KEY is index of qubit,
                            and VALUE denotes qubit's state (0 or 1)
        :return:
        """

        fixed_qubits = fixed_states.keys()

        circuit = []
        for qubit_index in range(self._number_of_qubits):
            if qubit_index in fixed_qubits:
                circuit.append(fixed_states[qubit_index])
            else:
                circuit.append(np.random.randint(0, 2))

        return circuit

    def add_circuits(self,
                     circuits: List[List[str]]):
        for circuit in circuits:
            self._circuits_list.append(circuit)

    def get_random_circuits(self,
                            number_of_circuits: int,
                            fixed_states: Optional[Dict[int, int]] = None):
        if fixed_states is None:
            return [self.get_random_circuit() for _ in range(number_of_circuits)]
        else:
            return [self.get_random_circuit_with_fixed_state_of_some_qubits(fixed_states) for _ in
                    range(number_of_circuits)]

    def add_random_circuits(self,
                            number_of_circuits: int,
                            fixed_states: Optional[Dict[int, int]] = None):

        self.add_circuits(self.get_random_circuits(number_of_circuits, fixed_states))

    @staticmethod
    def get_qubits_key(qubits_list: List[int]):
        return anf.get_qubits_key(qubits_list)
