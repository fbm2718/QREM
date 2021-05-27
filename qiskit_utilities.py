import numpy as np
from qiskit.result import Result
from typing import List
from povmtools import reorder_probabilities
import povmtools

def get_frequencies_array_from_results(results_list: List[Result]) -> np.ndarray:
    """
    Description:
        Creates an array of frequencies from given qiskit job results. This method_name is is working with
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

    # The length of potentially_stochastic_matrix state describes how many qubits were used during experiment.
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
            # frequencies = reorder_probabilities(normal_order, range(states_len))
            frequencies = reorder_probabilities(normal_order, range(states_len)[::-1])

            frequencies_array[i][:] = frequencies[:]

    return frequencies_array




def add_gate_to_circuit(circuit,
                        qreg,
                        q_index,
                        unitary):
    import QREM

    #TODO: Check whether this is needed - at some point I remember there were troubles with parametrizing those two unitaries
    if unitary[0,0]==1 and unitary[1,1] == 1:
        pass
    elif unitary[0,1]==1 and unitaryp[1,0] == 1:
        circuit.X(qreg[q_index])

    else:
        # get angles for single-qubit state change unitary
        current_angles = QREM.povmtools.get_su2_parametrizing_angles(unitary)

        # implement unitary
        circuit.u3(current_angles[0],
                   current_angles[1],
                   current_angles[2],
                   qreg[q_index])

    return circuit, qreg


def get_frequencies_from_counts(counts_dict,
                                crs=None,
                                possible_states = None,
                                shots_number = None,
                                reorder_bits = True):
    if crs is None:
        crs = len(list(list(counts_dict.keys())[0]))

    d = 2 ** crs

    if possible_states is None:
        possible_states = ["{0:b}".format(i).zfill(crs) for i in range(d)]

    normal_order = []

    for j in range(d):
        if possible_states[j] in counts_dict.keys():
            counts_now = counts_dict[possible_states[j]]
            normal_order.append(counts_now)

        else:
            normal_order.append(0)
    if reorder_bits:
       frequencies = reorder_probabilities(normal_order, range(crs)[::-1])
    else:
        frequencies = normal_order

    if shots_number is None:
        frequencies = frequencies/np.sum(frequencies)
    else:
        frequencies = frequencies/shots_number

    return frequencies



def save_counts_from_multiple_jobs(directory_data,backend_name,directory_save):
    from qiskit import IBMQ
    from povms_qi.qiskit_tools import backend_utilities, circuit_utilities
    from povms_qi import data_tools as dt
    from povms_qi import povm_data_tools as pdt
    from povms_qi import ancillary_functions as anf
    import os
    from tqdm import tqdm
    provider = IBMQ.get_provider(group='open')
    backend = provider.get_backend(backend_name)

    jobs_cluster = []

    files_inside = os.listdir(directory_data)

    # print(files_inside)
    # raise KeyError
    results_list = []
    for file_index in range(len(files_inside)):
        results_now = dt.open_file_simple(directory_data+files_inside[file_index])
        job_IDs_now = results_now['job_IDs']
        results_list.append(results_now)
        for job_index in range(len(job_IDs_now)):
            print('getting job number '+str(file_index)+str(job_index))
            job = backend.retrieve_job(job_IDs_now[job_index])
            print('got it\n')
            jobs_cluster.append(job)

    anf.cool_print('Getting counts:', 'starting...')
    all_counts = []

    for job_index in tqdm(range(len(jobs_cluster))):
        job_now = jobs_cluster[job_index]
        for exp_index in range(75):
            try:
                results_now = job_now.result()
                counts_now = results_now.get_counts(exp_index)
                all_counts.append(counts_now)
            except(IndexError):
                break
    anf.cool_print('Getting counts:', 'finished.')
    dictionary = {'results_list':results_list}
    dictionary['all_counts'] = all_counts
    # if saving:
    pdt.Save_Results_simple(dictionary,
                            directory_save,
                            'counts',
                            False)



def add_counts_dicts(all_counts,modulo,dimension):
    frequencies = [np.zeros(dimension) for i in range(modulo)]
    from tqdm import tqdm
    for counts_index in tqdm(range(len(all_counts))):
        true_index = counts_index%modulo

        freqs_now = povmtools.counts_dict_to_frequencies_vector(all_counts[counts_index],True)
        frequencies[true_index][:]+=freqs_now[:]

        # print(freqs_now)
    for i in range(modulo):
        frequencies[i]*=1/np.sum(frequencies[i])

    return frequencies


