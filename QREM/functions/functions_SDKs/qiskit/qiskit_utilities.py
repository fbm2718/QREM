"""
@authors: Filip Maciejewski, Oskar Słowik, Tomek Rybotycki
@contact: filip.b.maciejewski@gmail.com
"""
import numpy as np
from qiskit.result import Result
from typing import List, Dict
from functions.povmtools import reorder_probabilities
from functions import povmtools, ancillary_functions as anf
import time
from qiskit.providers.ibmq.job.ibmqjob import JobStatus
import qiskit

from qiskit import Aer, IBMQ
from tqdm import tqdm


def get_frequencies_array_from_results(results_list: List[Result]) -> np.ndarray:
    """
    Description:
        Creates an array of frequencies from given qiskit job results. This method_name is is working with
        qiskit 0.16. The shape of the array is
            c x 2 ** q,
        where c denotes circuits number and q denotes number of qubits.
    Parameters:
        :param results_list: List of qiskit jobs_list results.
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

    # TODO: Check whether this is needed - at some point I remember there were troubles with parametrizing those two unitaries
    if unitary[0, 0] == 1 and unitary[1, 1] == 1:
        pass
    elif unitary[0, 1] == 1 and unitary[1, 0] == 1:
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
                                possible_states=None,
                                shots_number=None,
                                reorder_bits=True):
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
        frequencies = frequencies / np.sum(frequencies)
    else:
        frequencies = frequencies / shots_number

    return frequencies


def download_multiple_jobs(backend_name,
                           job_IDs_list):
    IBMQ.load_account()
    if backend_name in ['qasm_simulator', 'statevector_simulator', 'unitary_simulator']:
        raise ValueError('Local simulators do not store jobs online.')
    else:
        provider = IBMQ.get_provider(group='open')
        backend = provider.get_backend(backend_name)

    all_jobs = []
    for job_ID in job_IDs_list:
        anf.cool_print('Getting job with ID:', job_ID)
        job = backend.retrieve_job(job_ID)
        anf.cool_print('Got it!')
        all_jobs.append(job)
    return all_jobs





def add_counts_dicts(all_counts, modulo, dimension):
    frequencies = [np.zeros(dimension) for i in range(modulo)]
    from tqdm import tqdm
    for counts_index in tqdm(range(len(all_counts))):
        true_index = counts_index % modulo

        freqs_now = povmtools.counts_dict_to_frequencies_vector(all_counts[counts_index], True)
        frequencies[true_index][:] += freqs_now[:]

        # print(freqs_now)
    for i in range(modulo):
        frequencies[i] *= 1 / np.sum(frequencies[i])

    return frequencies


def run_batches(batches,
                backend_name,
                shots=8192,
                saving_IDs_dictionary={'saving': False,
                                       'directory': None,
                                       'file_name': None,
                                       'dictionary_to_save': {}}):
    # IBMQ.load_account()
    # IBMQ.load_account()
    # raise KeyError
    saving = saving_IDs_dictionary['saving']

    anf.cool_print('\nSending jobs_list to execution on: ', backend_name + '.')
    anf.cool_print('Number of shots: ', str(shots) + ' .')
    anf.cool_print('Target number of jobs_list: ', str(len(batches)) + ' .')

    iterations_done = 0
    wait_time_in_minutes = 10

    print()
    jobs = []
    while iterations_done < len(batches):
        anf.cool_print('job number:', str(iterations_done))
        circuits = batches[iterations_done]

        if backend_name in ['qasm_simulator', 'statevector_simulator', 'unitary_simulator']:
            backend = Aer.get_backend(backend_name)
        else:
            IBMQ.load_account()
            provider = IBMQ.get_provider(group='open')
            backend = provider.get_backend(backend_name)

        try:
            time.sleep(2)
            qobj_id = 'first_circuit-' + circuits[0].name + '-last_circuit-' + circuits[
                -1].name + '-date-' + anf.get_date_string('-')
            anf.cool_print("Sending quantum program to: ", backend_name + '.')
            job = qiskit.execute(circuits, backend, shots=shots, max_credits=200, qobj_id=qobj_id)

            if saving and backend_name not in ['qasm_simulator',
                                               'statevector_simulator',
                                               'unitary_simulator']:
                job_ID = job.job_id()
                dict_to_save = saving_IDs_dictionary['dictionary_to_save']
                dict_to_save['job_ID'] = job_ID
                anf.save_results_pickle(dictionary_to_save=dict_to_save,
                                        directory=saving_IDs_dictionary['directory'],
                                        custom_name=saving_IDs_dictionary[
                                                        'file_name'] + '_job%s' % iterations_done)

            jobs.append(job)
            while job.status() == JobStatus.INITIALIZING:
                print(job.status())
                time.sleep(2)

            anf.cool_print("Program sent for execution to: ", backend_name + '.')

        except BaseException as ex:
            print('There was an error in the circuit!. Error = {}'.format(ex))
            print(f'Waiting {wait_time_in_minutes} minute(s) before next try.')
            time.sleep(wait_time_in_minutes * 60)
            continue

        print()
        iterations_done += 1

    return jobs


def get_counts_from_jobs(jobs_list,
                         return_job_headers=False) -> Dict[str, Dict[str, int]]:
    anf.cool_print('Getting counts...')

    counts_dictionary, job_headers = {}, []

    for job_index in tqdm(range(len(jobs_list))):
        job_now = jobs_list[job_index]
        job_header_now = job_now.result().qobj_id
        job_headers.append(job_header_now)

        results_object_now = job_now.result()
        results_list_now = list(results_object_now.results)
        for exp_index in range(len(results_list_now)):
            circuit_name_now = results_list_now[exp_index].header.name

            counts_now = results_object_now.get_counts(circuit_name_now)


            counts_dictionary[circuit_name_now] = counts_now

    if return_job_headers:
        return counts_dictionary, job_headers
    else:
        return counts_dictionary
