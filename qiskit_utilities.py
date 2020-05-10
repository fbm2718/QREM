import numpy as np
from qiskit.result import Result
from typing import List
from povmtools import reorder_probabilities


def get_frequencies_array_from_results(results_list: List[Result]) -> np.ndarray:
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

            frequencies = reorder_probabilities(normal_order, range(states_len)[::-1])

            frequencies_array[i][:] = frequencies[:]

    return frequencies_array
