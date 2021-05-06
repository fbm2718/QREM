"""
Created on 01.03.2021

@authors: Filip B. Maciejewski, Tomek Rybotycki, Oskar Słowik
@contact: filip.b.maciejewski@gmail.com
"""

import numpy as np
import copy
import QREM

from QREM import ancillary_functions as anf
from collections import defaultdict


class key_dependent_dict_for_marginals(defaultdict):
    def __init__(self):
        super().__init__(None)  # base class doesn't get a factory
        self.f_of_x = lambda key: np.zeros((int(2 ** len(anf.get_qubit_indices_from_string(key))), 1),
                                 dtype=float)  # save f(x)

    def __missing__(self, key):  # called when a default needed
        ret = self.f_of_x(key)  # calculate default value
        self[key] = ret  # and install it in the dict
        return ret




def get_state_from_circuit_name(circuit_name):
    state_name = ''
    for string in circuit_name:
        if string in ['1', 'X', 'x']:
            state_name += '1'
        elif string in ['0', 'I', 'i', 'id', 'Id']:
            state_name += '0'

    return state_name


def get_mini_dict(number_of_qubits):
    register = QREM.povmtools.register_names_qubits(range(number_of_qubits), number_of_qubits)
    return {key: np.zeros((int(2 ** number_of_qubits), 1)) for key in register}




def estimate_energy_from_marginals(weights_dictionary, marginals):
    energy = 0
    for key in weights_dictionary.keys():
        weight = weights_dictionary[key]
        marginal = marginals[key]

        # print(key, marginal)
        # raise KeyError
        qubits_number = int(np.log2(len(marginal)))

        for i in range(len(marginal)):
            bitstring = list(anf.binary_integer_format(i, qubits_number))
            bit_true = [int(x) for x in bitstring]
            parity = (-1) ** (np.count_nonzero(bit_true))
            energy += weight * marginal[i] * parity

    if isinstance(energy, list) or isinstance(energy, type(np.array(1))):
        return energy[0]
    else:
        return energy

