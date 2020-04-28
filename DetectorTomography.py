"""
Created on Wed Aug  8 21:40:15 2018

@author: Filip Maciejewski
email: filip.b.maciejewski@gmail.com

References:
[1] Z. Hradil, J. Řeháček, J. Fiurášek, and M. Ježek, “3 maximum-likelihood methods in quantum mechanics,” in Quantum
State Estimation, edited by M. Paris and J. Řeháček (Springer Berlin Heidelberg, Berlin, Heidelberg, 2004) pp. 59–112.
[2] J. Fiurášek, Physical Review A 64, 024102 (2001), arXiv:quant-ph/0101027 [quant-ph].
"""

import numpy as np
import scipy as sc
import copy
from math import log

from povmtools import get_density_matrix, permute_matrix, reorder_classical_register, sort_things, reorder_probabilities
from qiskit.result import Result
from typing import List

from PyMaLi.GeneralTensorCalculator import GeneralTensorCalculator


# GTC stands for General Tensor Calculator.
def gtc_tensor_counting_function(arguments: list):
    result = 1

    for a in arguments:
        result = np.kron(result, a)

    return result


def gtc_matrix_product_counting_function(arguments: list):
    result = arguments[0]

    i = 1

    while i < len(arguments):
        result = result @ arguments[i]
        i += 1

    return result


class DetectorTomographyFitter:
    """
        This class is meant to resemble qiskit's state tomography and process tomography fitters and to calculate the
        maximum likelihood povm estimator describing a detector basing on QDT job results and used probe states.
    """

    def __init__(self, algorithm_convergence_threshold=1e-6):
        self.algorithmConvergenceThreshold = algorithm_convergence_threshold

    def get_maximum_likelihood_povm_estimator(self, results_list: List[Result],
                                              probe_kets: List[np.array], qiskit_register_convention=True) \
            -> List[np.ndarray]:
        """
        Description:
            Given results of Quantum Detector Tomography experiments and list of probe states, return the Maximum
            Likelihood estimation of POVM describing a detector. Uses recursive method from [1]. See also [2].

        Parameters:
            :param results_list: List of results obtained from executing qiskit jobs.
            :param probe_kets: A set of probe kets used to perform tomography.
            :param qiskit_register_convention: Qiskit register convention is reverse to what we usually work with.
            This solves the problem.

        Returns
            Maximum likelihood estimator of POVM describing a detector.
        """

        frequencies_array = self.__get_frequencies_array_from_results(results_list, qiskit_register_convention)

        # FROM THIS MOMENT IT WILL BE REMOVED WHEN GENERAL METHODS ARE CREATED!
        probe_states = self.__get_probe_states(results_list, probe_kets)
        number_of_probe_states = frequencies_array.shape[1]
        dimension = probe_states[0].shape[0]

        povm = []

        for j in range(number_of_probe_states):
            povm.append(np.identity(dimension) / number_of_probe_states)

        # Threshold is dynamic, thus another variable
        threshold = self.algorithmConvergenceThreshold

        i = 0
        current_difference = 1
        while abs(current_difference) >= threshold:
            i += 1

            if i % 50 == 0:
                last_step_povm = copy.copy(povm)

            r_matrices = [self.__get_r_operator(povm[j], j, frequencies_array, probe_states)
                          for j in range(number_of_probe_states)]
            lagrange_matrix = self.__get_lagrange_matrix(r_matrices, povm)
            povm = [self.__calculate_symmetric_m(lagrange_matrix, r_matrices[j], povm[j])
                    for j in range(number_of_probe_states)]

            if i % 50 == 0:  # calculate the convergence test only sometimes to make the code faster
                current_difference = sum([np.linalg.norm(povm[k] - last_step_povm[k], ord=2)
                                          for k in range(number_of_probe_states)])

            elif i > 5e5:  # make sure it does not take too long, sometimes convergence might not be so good
                threshold = 1e-3

            elif i > 1e5:  # make sure it does not take too long, sometimes convergence might not be so good
                threshold = 1e-4

        return povm

    def __get_maximum_likelihood_povm_from_frequencies(self, frequencies_array: np.ndarray,
                                                       probe_kets: List[np.array]) \
            -> List[np.ndarray]:

        probe_states = self.__get_probe_states(results_list, probe_kets)
        number_of_probe_states = frequencies_array.shape[1]
        dimension = probe_states[0].shape[0]

        povm = []

        for j in range(number_of_probe_states):
            povm.append(np.identity(dimension) / number_of_probe_states)

        # Threshold is dynamic, thus another variable
        threshold = self.algorithmConvergenceThreshold

        i = 0
        current_difference = 1
        while abs(current_difference) >= threshold:
            i += 1

            if i % 50 == 0:
                last_step_povm = copy.copy(povm)

            r_matrices = [self.__get_r_operator(povm[j], j, frequencies_array, probe_states)
                          for j in range(number_of_probe_states)]
            lagrange_matrix = self.__get_lagrange_matrix(r_matrices, povm)
            povm = [self.__calculate_symmetric_m(lagrange_matrix, r_matrices[j], povm[j])
                    for j in range(number_of_probe_states)]

            if i % 50 == 0:  # calculate the convergence test only sometimes to make the code faster
                current_difference = sum([np.linalg.norm(povm[k] - last_step_povm[k], ord=2)
                                          for k in range(number_of_probe_states)])

            elif i > 5e5:  # make sure it does not take too long, sometimes convergence might not be so good
                threshold = 1e-3

            elif i > 1e5:  # make sure it does not take too long, sometimes convergence might not be so good
                threshold = 1e-4

        return povm

    @staticmethod
    def __get_probe_states(results_list: List[Result],
                           probe_kets: List[np.array]) -> List[np.ndarray]:
        """
        Description:
            This method generates probe states (density matrix) from results and kets
            passed to maximum likelihood POVM counting object.
        Parameters:
            :param results_list: List of results obtained from qiskit jobs execution.
            :param probe_kets: Kets on which job circuits were based.
        Returns:
            List of probe states. These are supposed to have dimension equal to the size of Hilbert space, hence if one
            have used tensor products of single-qubit states, then one needs to give here those tensor products. Order
            needs to fit this of results.results.
        """

        circuits_number = sum(len(results.results) for results in results_list)

        # This is a little elaborate, but necessary.
        qubits_number = int(log(circuits_number, len(probe_kets)))

        probe_states = []

        for i in range(qubits_number):
            probe_states.append([get_density_matrix(ket) for ket in probe_kets])

        general_tensor_calculator = GeneralTensorCalculator(gtc_tensor_counting_function)

        return general_tensor_calculator.calculate_tensor_to_increasing_list(probe_states)

    @staticmethod
    def __get_frequencies_array_from_results(results_list: List[Result], qiskit_register_convention) -> np.ndarray:
        """
        Description:
            Creates an array of frequencies from given qiskit job results. This method is is working with
            qiskit 0.16. The shape of the array is

                c x 2 ** q,

            where c denotes circuits number and q denotes number of qubits.

        Parameters:
            :param results_list: List of qiskit jobs results.

        Returns:
            np.ndarray with shape=0 if there were no circuits in the job, or with shape c x 2 ** q
            containing frequencies data for each possible state.

        Notes:
            Possible states are numbered increasingly from |00000 ... 0>, |10000 ... 0> up to |1111 ... 1>.
        """

        all_circuits_number = sum(len(results.results) for results in results_list)

        if all_circuits_number == 0:
            return np.ndarray(shape=0)

        # The length of a state describes how many qubits were used during experiment.
        states_len = len(next(iter(results_list[0].get_counts(0).keys())))

        possible_states = ["{0:b}".format(i).zfill(states_len) for i in range(2 ** states_len)]
        frequencies_array = np.ndarray(shape=(all_circuits_number, len(possible_states)))

        # TODO TR: This has to be rewritten as it's too nested.
        for results in results_list:
            number_of_circuits_in_results = len(results.results)
            for i in range(number_of_circuits_in_results):
                counts = results.get_counts(i)
                shots_number = results.results[i].shots

                # TODO FBM: added here accounting for reversed register in qiskit
                normal_order = []
                for j in range(len(possible_states)):
                    if possible_states[j] in counts.keys():
                        normal_order.append(counts[possible_states[j]] / shots_number)
                    else:
                        normal_order.append(0)

                if qiskit_register_convention:
                    frequencies = reorder_probabilities(normal_order, range(states_len)[::-1])
                else:
                    frequencies = normal_order

                frequencies_array[i][:] = frequencies[:]

        return frequencies_array

    @staticmethod
    def __get_r_operator(m_m: np.ndarray, index_of_povm_effect: int, frequencies_array: np.ndarray,
                         probe_states: List[np.ndarray]) -> np.ndarray:
        """
        Description:
            This method calculates R operator as defined in Ref. [1].

        Parameters:
            :param m_m: Effect for which R operator is calculated.
            :param index_of_povm_effect: Index of povm effect for which R is calculated.
            :param frequencies_array: frequencies_array - array with size (m x n), where m means number of probe states,
            n means number of POSSIBLE outcomes.
            :param probe_states: A list of probe states density matrices.

        Returns:
            The R operator as described in Ref. [1].
        """

        number_of_probe_states = frequencies_array.shape[0]

        d = probe_states[0].shape[0]

        m_r = np.zeros((d, d), dtype=complex)

        for probe_state_index in range(number_of_probe_states):
            expectation_value_on_probe_state = np.trace(m_m @ probe_states[probe_state_index])

            if expectation_value_on_probe_state == 0j:
                continue

            x = frequencies_array[probe_state_index, index_of_povm_effect] / expectation_value_on_probe_state
            m_r += x * probe_states[probe_state_index]

        if np.linalg.norm(m_r) == 0:
            m_r = np.zeros((d, d), dtype=complex)

        return m_r

    @staticmethod
    def __get_lagrange_matrix(r_matrices: List[np.ndarray], povms: List[np.ndarray]) -> np.ndarray:
        """
        Description:
            Calculates Lagrange matrix used in Lagrange multipliers optimization method.

        Parameters:
            :param r_matrices: A list of R matrices described in a method generating them.
            :param povms: A list of effects for which Lagrange matrix will be calculated.

        Returns:
           Lagrange matrix for given parameters.
        """
        number_of_povms = len(povms)
        dimension = povms[0].shape[0]
        second_power_of_lagrange_matrix = np.zeros((dimension, dimension), dtype=complex)

        for j in range(number_of_povms):
            second_power_of_lagrange_matrix += r_matrices[j] @ povms[j] @ r_matrices[j]

        lagrange_matrix = sc.linalg.sqrtm(second_power_of_lagrange_matrix)

        return lagrange_matrix

    @staticmethod
    def __calculate_symmetric_m(m_lagrange_matrix: np.ndarray, m_r: np.ndarray, m_m: np.ndarray) -> np.ndarray:
        """
        Description:
            A method used for calculating symmetric m matrix.

        Parameters:
            :param m_m: A matrix of which symmetric version will be calculated.
            :param m_r: Previously calculated R operator.
            :param m_lagrange_matrix:

        Returns:
            Symmetric m matrix.
        """
        try:
            # Try to perform inversion of lagrange matrix.
            m_inversed_lagrange_matrix = np.linalg.inv(m_lagrange_matrix)
        except np.linalg.LinAlgError:
            # In some special cases it may fail. Provide identity matrix in that scenario.
            m_inversed_lagrange_matrix = np.eye(np.shape(m_lagrange_matrix)[0])

        symmetric_m = m_inversed_lagrange_matrix @ m_r @ m_m @ m_r @ m_inversed_lagrange_matrix

        return symmetric_m

    # TODO TR: This method may need to be revisited and possibly reduced into several smaller ones.
    @staticmethod
    def join_povms(povms: List[List[np.ndarray]], qubit_indices_lists: List[List[int]]) -> List[np.ndarray]:
        """
        Description:
            Generates a POVM from given list of POVMs and qubit indices.

        Parameter:
            :param povms: List of POVMs corresponding to qubits indices.
            :param qubit_indices_lists: Indices of qubits for which POVMs were calculated.

        Return:
            POVM describing whole detector.
        """
        qubits_num = sum([len(indices) for indices in qubit_indices_lists])

        swapped_povms = []

        for i in range(len(qubit_indices_lists)):
            indices_now = qubit_indices_lists[i]
            povm_now = povms[i]

            # extend povm to higher-dimensional space.by multiplying by complementing identity from right
            povm_now_extended = [np.kron(Mi, np.eye(2 ** (qubits_num - len(indices_now)))) for Mi in povm_now]

            # begin swapping
            swapped_povm = copy.copy(povm_now_extended)

            # go from back to ensure commuting
            for j in range(len(indices_now))[::-1]:
                index_qubit_now = indices_now[j]

                if index_qubit_now != j:
                    # swap qubit from jth position to proper position
                    swapped_povm = [permute_matrix(Mj, qubits_num, (j + 1, index_qubit_now + 1)) for Mj in swapped_povm]

            swapped_povms.append(swapped_povm)

        # With POVMs now represented by proper matrices (that is, parts corresponding to adequate qubits are now in
        # desired places), we want to join them.
        general_tensor_calculator = GeneralTensorCalculator(gtc_matrix_product_counting_function)
        povm = general_tensor_calculator.calculate_tensor_to_increasing_list(swapped_povms)

        # We've obtained a POVM, but it is still ordered according to qubit indices. We want to undo that.
        indices_order = []
        for indices_list in qubit_indices_lists:
            indices_order = indices_order + indices_list

        new_classical_register = reorder_classical_register(indices_order)
        sorted_povm = sort_things(povm, new_classical_register)

        return sorted_povm


class QDTCalibrationSetup:

    def __init__(self, qubits_number: int, ):
        self.qubits_number = qubits_number
