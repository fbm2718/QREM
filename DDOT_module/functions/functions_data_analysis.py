"""
Created on 01.03.2021

@authors: Filip B. Maciejewski, Tomek Rybotycki, Oskar Słowik
@contact: filip.b.maciejewski@gmail.com
"""

from collections import defaultdict
from typing import Dict

import numpy as np
from QREM import ancillary_functions as anf
from QREM import povmtools


class KeyDependentDictForMarginals(defaultdict):
    """
    This is class used to store marginal probability distributions in dictionary.
    It is "key dependent" in the sense that if user tries to refer to non-existing value for some
    KEY, then this value is created as potentially_stochastic_matrix marginal distribution which size depends on the KEY
    NOTE: We assume that provided KEY is potentially_stochastic_matrix string denoting  qubits subset
    (see self.value_creating_function)


    COPYRIGHT NOTE
    The main idea of this code was taken from Reddit thread:
    https://www.reddit.com/r/Python/comments/27crqg/making_defaultdict_create_defaults_that_are_a/

    """

    def __init__(self):
        super().__init__(None)  # initialize as standard defaultdict

        # This is the function which takes the string "key" that is assumed to label qubits subset
        # in the form 'q2q3q11...' etc. It takes this key, calculates number of qubits N, and creates
        # empty vector of the size d=2^N.
        self.value_creating_function = lambda key: np.zeros(
            (int(2 ** len(anf.get_qubit_indices_from_string(key))), 1),
            dtype=float)

    # called when key is missing
    def __missing__(self, key):
        # calculate the key-dependent value
        ret = self.value_creating_function(key)
        # put the value inside the dictionary
        self[key] = ret
        return ret


class key_dependent_dict_for_marginals(defaultdict):
    """
    same as KeyDependentDictForMarginals but different name
    TODO FBM: refactor this
    """

    def __init__(self):
        super().__init__(None)  # initialize as standard defaultdict

        # This is the function which takes the string "key" that is assumed to label qubits subset
        # in the form 'q2q3q11...' etc. It takes this key, calculates number of qubits N, and creates
        # empty vector of the size d=2^N.
        self.value_creating_function = lambda key: np.zeros(
            (int(2 ** len(anf.get_qubit_indices_from_string(key))), 1),
            dtype=float)

    # called when key is missing
    def __missing__(self, key):
        # calculate the key-dependent value
        ret = self.value_creating_function(key)
        # put the value inside the dictionary
        self[key] = ret
        return ret

def get_state_from_circuit_name(circuit_name):
    state_name = ''
    for string in circuit_name:
        if string in ['1', 'X', 'x']:
            state_name += '1'
        elif string in ['0', 'I', 'i_index', 'id', 'Id']:
            state_name += '0'

    return state_name


def get_mini_dict(number_of_qubits):
    register = povmtools.register_names_qubits(range(number_of_qubits), number_of_qubits)
    return {key: np.zeros((int(2 ** number_of_qubits), 1)) for key in register}


def estimate_energy_from_marginals(weights_dictionary:Dict[str,float],
                                   marginals_dictionary:Dict[str,np.ndarray]):
    """
    Compute energy of Hamiltonian from dictionary of marginal distributions.

    :param weights_dictionary:
    :param marginals_dictionary:
    :return:
    """

    energy = 0
    for key_local_term in weights_dictionary.keys():
        weight = weights_dictionary[key_local_term]
        marginal = marginals_dictionary[key_local_term]

        qubits_number = int(np.log2(len(marginal)))

        for result_index in range(len(marginal)):
            bitstring = list(anf.binary_integer_format(result_index, qubits_number))
            bit_true = [int(x) for x in bitstring]
            parity = (-1) ** (np.count_nonzero(bit_true))
            energy += weight * marginal[result_index] * parity

    if isinstance(energy, list) or isinstance(energy, np.ndarray):
        return energy[0]
    else:
        return energy
