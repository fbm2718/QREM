"""
@authors: Filip Maciejewski, Oskar Słowik, Tomek Rybotycki
@contact: filip.b.maciejewski@gmail.com
"""
import cmath as c
import copy
import datetime as dt
import itertools
import os
import pickle
import re
from collections import defaultdict
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from colorama import Fore, Style

epsilon = 10 ** (-7)
pauli_sigmas = {
    'id': np.array([[1., 0j], [0, 1]]),
    'X': np.array([[0.j, 1], [1, 0]]),
    'Y': np.array([[0., -1j], [1j, 0]]),
    'Z': np.array([[1., 0j], [0, -1]])
}
___bell_states___ = {'phi+': 1 / np.sqrt(2) * np.array([1, 0, 0, 1]),
                     'phi-': 1 / np.sqrt(2) * np.array([1, 0, 0, -1]),
                     'psi+': 1 / np.sqrt(2) * np.array([0, 1, 1, 0]),
                     'psi-': 1 / np.sqrt(2) * np.array([0, 1, -1, 0]), }

__alphabet__ = ['']

__standard_gates = {'id': np.array([[1., 0j], [0, 1]], dtype=complex),
                    'X': np.array([[0.j, 1], [1, 0]], dtype=complex),
                    'Y': np.array([[0., -1j], [1j, 0]]),
                    'Z': np.array([[1., 0j], [0, -1]], dtype=complex),
                    'S': np.array([[1, 0], [0, 1j]]),
                    'T': np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]]),
                    'H': 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=complex)}

__pauli_eigenkets__ = {'x+': 1 / np.sqrt(2)*np.array([[1], [1]], dtype=complex),
                       'x-': 1 / np.sqrt(2)*np.array([[1], [-1]], dtype=complex),
                       'y+': 1 / np.sqrt(2)*np.array([[1], [1j]]),
                       'y-': 1 / np.sqrt(2)*np.array([[1], [-1j]]),
                       'z+': np.array([[1], [0]]),
                       'z-': np.array([[0], [1]])}


def integer_representation(integer: int,
                           base: int):
    if integer > base:
        raise ValueError('Integer bigger than base!')

    if base < 10:
        return integer
    else:
        if integer < 10:
            return integer
        else:
            return chr(integer + 97 - 10)


def integer_representation_above10(integer: int):
    if integer < 10:
        return str(integer)
    else:
        return chr(integer + 97 - 10)


def get_bell_basis():
    return list(___bell_states___.values())


class key_dependent_dict(defaultdict):
    """
    This is class used to construct dictionary which creates values of keys in situ, in case there
    user refers to key that is not present.

    COPYRIGHT NOTE
    This code was taken from Reddit thread:
    https://www.reddit.com/r/Python/comments/27crqg/making_defaultdict_create_defaults_that_are_a/

    """

    def __init__(self, f_of_x=None):
        super().__init__(None)  # base class doesn't get potentially_stochastic_matrix factory
        self.f_of_x = f_of_x  # save f(x)

    def __missing__(self, key):  # called when potentially_stochastic_matrix default needed
        ret = self.f_of_x(key)  # calculate default value
        self[key] = ret  # and install it in the dict
        return ret


def is_stochastic(potentially_stochastic_matrix: np.ndarray,
                  stochasticity_type: Optional[str] = 'left') -> bool:
    """
    :param potentially_stochastic_matrix:
    :param stochasticity_type: string specyfing what type of stochasticity we want to tests
    possible options:

    - 'left' - columns are probability distributions
    - 'right' - rows are probability distributions
    - 'doubly' or 'ortho' - both columns and rows are probability distributions

    :return:
    """
    shape = potentially_stochastic_matrix.shape[0]

    if stochasticity_type == 'left':
        for index_row in range(shape):
            one_now = sum(potentially_stochastic_matrix[:, index_row])

            if abs(1 - one_now) >= 10 ** (-6):
                return False
    elif stochasticity_type == 'right':
        for index_row in range(shape):
            one_now = sum(potentially_stochastic_matrix[index_row, :])

            if abs(1 - one_now) >= 10 ** (-6):
                return False
    elif stochasticity_type == 'ortho' or stochasticity_type == 'doubly':
        for index_both in range(shape):
            one_now = sum(potentially_stochastic_matrix[:, index_both])
            one_now2 = sum(potentially_stochastic_matrix[index_both, :])
            if abs(1 - one_now) >= 10 ** (-6) or abs(1 - one_now2) >= 10 ** (-6):
                return False

    else:
        raise ValueError('Wrong stochasticity_type of stochasticity')

    return True


def binary_integer_format(integer: int,
                          number_of_bits: int) -> str:
    """
    Return binary representation of an integer
    :param integer:
    :param number_of_bits:
    NOTE: number of bits can be greater than minimal needed to represent integer
    :return:
    """
    return "{0:b}".format(integer).zfill(number_of_bits)


# TODO FBM: think whether the difference between this function and next matters
# def enumerated_dictionary(some_list):
#     return dict((sorted_i, true_j) for sorted_i, true_j in enumerate(some_list))

def enumerated_dictionary(some_list):
    return dict(enumerate(some_list))


def get_reversed_enumerated_from_dict(enumerated_dict: Dict[int, int]) -> Dict[int, int]:
    """
    Get inverse of enumerated dictionary.
    :param enumerated_dict:
    :return:
    """
    reversed_map = {}
    for index_sorted, true_index in enumerated_dict.items():
        reversed_map[true_index] = index_sorted
    return reversed_map


def get_reversed_enumerated_from_indices(indices: List[int]) -> Dict[str, int]:
    """
    Given indices, return map which is inverse of enumerate
    :param indices:
    :return:
    """
    return get_reversed_enumerated_from_dict(enumerated_dictionary(indices))


def get_qubit_indices_from_string(qubits_string: str,
                                  with_q: Optional[bool] = False) -> List[int]:
    """Return list of qubit indices from the string of the form "q0q1q22q31"
    :param qubits_string (string): string which has the form of "q" followed by qubit index
    :param (optional) with_q (Boolean): specify whether returned indices
                                        should be in form of string with letter

    :return: list of qubit indices:

    depending on value of parameter "with_q" the mapping will be one of the following:

    if with_q:
        'q1q5q13' -> ['q1','q5','q13']
    else:
        'q1q5q13' -> [1,5,13]
    """

    numbers = re.findall(r'\d+', qubits_string)

    if with_q:
        qubits = ['q' + s for s in numbers]
    else:
        qubits = [int(s) for s in numbers]

    return qubits


def get_qubits_key(list_of_qubits: List[int]) -> str:
    """ from subset of qubit indices get the string that labels this subset
        using convention 'q5q6q12...' etc.
    :param list_of_qubits: labels of qubits

    :return: string label for qubits

     NOTE: this function is "dual" to get_qubit_indices_from_string.
    """

    return 'q' + 'q'.join([str(s) for s in list_of_qubits])


def round_matrix(matrix_to_be_rounded: np.ndarray,
                 decimal: int) -> np.ndarray:
    """
    This function rounds matrix in a nice way.
    "Nice" means that it removes funny Python artifacts such as "-0.", "0j" etc.

    :param matrix_to_be_rounded:
    :param decimal:
    :return:
    """

    data_type = type(matrix_to_be_rounded[0, 0])

    first_dimension = matrix_to_be_rounded.shape[0]
    second_dimension = np.size(matrix_to_be_rounded, axis=1)

    rounded_matrix = np.zeros((first_dimension, second_dimension), dtype=data_type)

    for first_index in range(0, first_dimension):
        for second_index in range(0, second_dimension):
            real_part = round(np.real(matrix_to_be_rounded[first_index, second_index]), decimal)

            if data_type == complex or data_type == np.complex128:
                imaginary_part = round(np.imag(matrix_to_be_rounded[first_index, second_index]),
                                       decimal)
            else:
                imaginary_part = 0

            # In the following we check whether some parts are 0 and then we leave it as 0
            # Intention here is to remove some Python artifacts such as leaving "-0" instead of 0.
            # see function's description.
            if abs(real_part) != 0 and abs(imaginary_part) != 0:
                if data_type == complex or data_type == np.complex128:
                    rounded_matrix[first_index, second_index] = real_part + 1j * imaginary_part
                else:
                    rounded_matrix[first_index, second_index] = real_part
            elif abs(real_part) == 0 and abs(imaginary_part) == 0:
                rounded_matrix[first_index, second_index] = 0
            elif abs(real_part) == 0 and abs(imaginary_part) != 0:
                if data_type == complex or data_type == np.complex128:
                    rounded_matrix[first_index, second_index] = 1j * imaginary_part
                else:
                    rounded_matrix[first_index, second_index] = 0
            elif abs(real_part) != 0 and abs(imaginary_part) == 0:
                rounded_matrix[first_index, second_index] = real_part

    return rounded_matrix


# TODO TR: This function should be renamed.
def sandwich(matrix_to_be_rotated, unitary_operator):
    return unitary_operator @ matrix_to_be_rotated @ np.matrix.getH(unitary_operator)


# ===========================================================================

# ===========================================================================
def zero_check(potential_zero_matrix,
               threshold=epsilon,
               method='numpy'):
    """
    Functions that checks if matrix is zero.
    :param potential_zero_matrix:
    :param threshold:
    :return:
    """

    size = list(potential_zero_matrix.shape)

    if method == 'numpy':
        zeros = np.zeros(size, dtype=type(potential_zero_matrix[0, 0]))
        return np.allclose(potential_zero_matrix, zeros, rtol=threshold)

    elif method == 'bruteforce':
        m_b = copy.deepcopy(potential_zero_matrix)
        if threshold >= 1:
            decimal = threshold
            threshold = 10 ** (-decimal)

        if len(size) == 1:
            m_b = potential_zero_matrix.reshape(size[0], 1)
            size.append(1)

        for i in range(0, size[0]):
            for j in range(0, size[1]):
                check_zero = np.abs(m_b[i, j])
                if check_zero > threshold:
                    return False
        return True


def calculate_outer_product(ket):
    return ket @ np.matrix.getH(ket)


# TODO TR: Check whether variables has been properly renamed.
def spectral_decomposition(matrix: np.ndarray):
    eigen_values, eigen_vectors = np.linalg.eig(matrix)

    dimension = matrix.shape[0]
    projectors = [calculate_outer_product(np.array(eigen_vectors[:, i]).reshape(dimension, 1)) for i in
                  range(dimension)]

    return eigen_values, projectors


# =======================================================================================
# Function that checks if matrix is identity. If so, it returns True, otherwise - False.
# =======================================================================================
def identity_check(m_a, eps=epsilon):
    size = np.size(m_a, 0)
    m_b = thresh(m_a)

    m_b_phase = c.phase(m_b[0, 0])
    m_b_prime = c.exp(-1j * m_b_phase) * m_b

    identity_matrix = np.identity(size)

    checking_zeros = thresh(m_b_prime - identity_matrix)

    return True if zero_check(checking_zeros, eps) else False


# TODO TR: Comment this method_name, and intention behind it.
def thresh(m_a, decimal=7):
    m_b = np.array(copy.deepcopy(m_a))
    with np.nditer(m_b, op_flags=['readwrite']) as it:
        for x in it:
            x[...] = np.round(x, decimal)

    return m_b


def lists_intersection(lst1: list,
                       lst2: list):
    return list(set(lst1) & set(lst2))


def lists_difference(lst1: list,
                     lst2: list):
    return list(set(lst1) - set(lst2))


def lists_sum_multi(lists: List[list]):
    return set().union(*lists)


def lists_sum(lst1: list,
              lst2: list):
    return list(set(lst1).union(set(lst2)))


def lists_intersection_multi(lists: List[list]):
    l0 = lists[0]
    l1 = lists[1]

    int_list = lists_intersection(l0, l1)
    for l in lists[2:]:
        int_list = lists_intersection(int_list, l)
    return int_list


def check_if_there_are_common_elements(lists: List[list]):
    for i in range(len(lists)):
        for j in range(i + 1, len(lists)):
            if len(lists_intersection(lists[i], lists[j])) != 0:
                return True

    return False


def find_significant_digit(x):
    if x == np.Inf or x == np.nan:
        return 0

    counter = 0
    passed = 0

    y = str(x)

    # print(x)
    if (y[0] == '-'):
        y.replace('-', '')

    for k in range(len(y)):
        if (y[k] == '.'):
            passed = 1
            dec = k

        if (passed == 1 and y[k] != '.'):
            if (y[k] == '0'):
                counter += 1
            else:
                counter += 1
                break

    if (len(y) - 2 == dec and y[len(y) - 1] == '0'):
        counter = 0

    return counter


def cool_print(colored_string: str,
               stuff_to_print: Optional = '',
               color=Fore.CYAN) -> None:
    """

    :param colored_string:  is printed with color
    :param stuff_to_print: is printed without color
    :param color:
    :return:
    """

    if isinstance(color, str):
        if color in ['red', 'RED', 'Red']:
            color = Fore.RED
        elif color in ['green', 'GREEN', 'Green']:
            color = Fore.BLUE
        elif color in ['blue', 'BLUE', 'Blue']:
            color = Fore.GREEN

    if stuff_to_print == '':
        print(color + Style.BRIGHT + str(colored_string) + Style.RESET_ALL)
    elif stuff_to_print == '\n':
        print(color + Style.BRIGHT + str(colored_string) + Style.RESET_ALL)
        print()
    else:
        print(color + Style.BRIGHT + str(colored_string) + Style.RESET_ALL, repr(stuff_to_print))


def bit_strings(number_of_qubits: int,
                reversed: Optional[bool] = False):
    """Generate outcome bitstrings for n-qubits.

    Args:
        number_of_qubits (int): the number of qubits.

    Returns:
        list: arrray_to_print list of bitstrings ordered as follows:
        Example: n=2 returns ['00', '01', '10', '11'].
"""
    if (reversed == True):
        return [(bin(j)[2:].zfill(number_of_qubits))[::-1] for j in list(range(2 ** number_of_qubits))]
    else:
        return [(bin(j)[2:].zfill(number_of_qubits)) for j in list(range(2 ** number_of_qubits))]


def register_names_qubits(qubit_indices,
                          quantum_register_size=None,
                          rev=False):
    """
    Register of qubits of size quantum_register_size, with only bits corresponding to qubit_indices
    varying

    :param qubit_indices:
    :param quantum_register_size:
    :param rev:
    :return:
    """

    # TODO FBM: refactor this function.

    if quantum_register_size is None:
        quantum_register_size = len(qubit_indices)

    if quantum_register_size == 0:
        return ['']

    if (quantum_register_size == 1):
        return ['0', '1']

    all_names = bit_strings(quantum_register_size, rev)
    not_used = []

    for j in list(range(quantum_register_size)):
        if j not in qubit_indices:
            not_used.append(j)

    bad_names = []
    for name in all_names:
        for k in (not_used):
            rev_name = name[::-1]
            if (rev_name[k] == '1'):
                bad_names.append(name)

    relevant_names = []
    for name in all_names:
        if name not in bad_names:
            relevant_names.append(name)

    return relevant_names


def get_module_directory():
    from QREM import name_holder
    name_holder = name_holder.__file__

    return name_holder[0:-15]


def zeros_to_dots(matrix,
                  decimal):
    m = matrix.shape[0]
    n = matrix.shape[1]

    B = np.zeros((m, n), dtype=dict)

    for i in range(m):
        for j in range(n):
            el = matrix[i, j]
            if (abs(np.round(el, decimal)) >= 0):
                B[i, j] = el
            else:
                B[i, j] = '.'

    return B


def print_array_nicely(arrray_to_print,
                       rounding_decimal=3):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 110)
    try:
        if (arrray_to_print.shape[0] == 1 or arrray_to_print.shape[1] == 1):
            B = copy.deepcopy(arrray_to_print)
            if (arrray_to_print.shape[0] == 1 and arrray_to_print.shape[1] == 1):
                print(np.round(arrray_to_print[0, 0], rounding_decimal))
            elif (arrray_to_print.shape[0] == 1):
                print([np.round(x[1], rounding_decimal) for x in arrray_to_print])
            elif (arrray_to_print.shape[1] == 1):
                print([np.round(x[0], rounding_decimal) for x in arrray_to_print])
        else:
            B = copy.deepcopy(arrray_to_print)
            C = round_matrix(B, rounding_decimal)
            D = zeros_to_dots(C, rounding_decimal)
            print(pd.DataFrame(D))
    except(IndexError):
        if len(arrray_to_print.shape) == 1:
            print([np.round(x, rounding_decimal) for x in arrray_to_print])
        else:
            print(pd.DataFrame(np.array(np.round(arrray_to_print, rounding_decimal))))


def gate_proper_date_string():
    ct0 = str(dt.datetime.today())
    ct1 = str.replace(ct0, ':', '_')
    ct2 = str.replace(ct1, '.', '_')
    return ct2[0:19]


def open_file_pickle(file_path):
    with open(file_path, 'rb') as filein:
        data_object = pickle.load(filein)

    return data_object


def save_results_pickle(dictionary_to_save,
                        directory,
                        custom_name: Optional[str] = 'no',
                        get_cwd=False):
    if (directory != None):
        fp0 = [s for s in directory]

        if (fp0[len(fp0) - 1] != '/'):
            fp0.append('/')
        fp = ''.join(fp0)


    else:
        fp = ''
    # Time& date
    if (get_cwd):
        cwd = os.getcwd()
    else:
        cwd = ''
    ct0 = str(dt.datetime.today())
    ct1 = str.replace(ct0, ':', '_')
    ct2 = str.replace(ct1, '.', '_')
    ct3 = ct2[0:19]
    # original_umask = os.umask(0)
    # os.umask(original_umask)

    main_directory = cwd + fp

    # permission_mode=int('0777',8)
    # os.chmod(main_directory,permission_mode)
    check_dir = os.path.exists(main_directory)

    if (check_dir == False):
        # print()
        #
        # # oldmask = os.umask(000)
        # print(main_directory)

        try:
            os.makedirs(main_directory)

        except(FileExistsError):
            import shutil
            try:
                shutil.rmtree(main_directory)
                os.makedirs(main_directory)
            except(FileExistsError):
                os.makedirs(main_directory)

        print(
            Fore.CYAN + Style.BRIGHT + 'Attention: ' + Style.RESET_ALL + 'Directory ' + '"' + main_directory + '"' + ' was created.')
        # os.umask(oldmask)
        # os.chmod(main_directory,permission_mode)
    if (custom_name == 'no'):
        file_path = str(main_directory + 'Results_Object' + '___' + str(ct3))
    else:
        file_path = str(main_directory + custom_name)

    # os.chmod(main_directory)
    dict_results = dictionary_to_save

    add_end = ''

    if (file_path[len(file_path) - 4:len(file_path)] != '.pkl'):
        add_end = '.pkl'

    # os.chmod(file_path)
    with open(file_path + add_end, 'wb') as f:
        pickle.dump(dict_results, f, pickle.HIGHEST_PROTOCOL)


def get_k_local_subsets(number_of_qubits,
                        locality):
    return list(itertools.combinations(range(number_of_qubits), locality))


def query_yes_no(question):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    yes_answers = {'yes',
                   'y',
                   'ye',
                   'tak',
                   'sure',
                   'of course',
                   'Yes',
                   'yeah'}
    no_answers = {'no',
                  'n',
                  'nope',
                  'nah',
                  'nie',
                  'noo',
                  'nooo',
                  'noooo',
                  'No'}

    existential_answers = {'I am never sure about anything', 'What is certain in this world?'}
    bad_answers = {'fuck', 'shit', 'dupa'}
    choice = 0
    print(question + ' [y/n]')
    choice = input().lower()
    if choice in yes_answers:
        return True
    elif choice in no_answers:
        return False
    else:
        if choice in existential_answers:
            cool_print('I feel you. However:', '')
        if choice in bad_answers:
            cool_print('Oh come on, how old are you?!', '')
        cool_print('Please:', "respond with 'yes' or 'no'")
        return query_yes_no(question)


def get_date_string(connector='_'):
    ct0 = str(dt.datetime.today())

    ct1 = str.replace(ct0, ':', connector)
    ct2 = str.replace(ct1, '.', connector)
    ct3 = ct2[0:19]

    return ct3
