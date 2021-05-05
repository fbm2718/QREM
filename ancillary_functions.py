"""
Created on Fri Feb 23 00:06:42 2018

@author: Filip Maciejewski
"""

import numpy as np
import cmath as c
import copy
import re
from colorama import Fore, Style
from collections import defaultdict

epsilon = 10 ** (-7)
pauli_sigmas = {
    'id': np.array([[1., 0j], [0, 1]]),
    'X': np.array([[0.j, 1], [1, 0]]),
    'Y': np.array([[0., -1j], [1j, 0]]),
    'Z': np.array([[1., 0j], [0, -1]])
}


class key_dependent_dict(defaultdict):
    def __init__(self, f_of_x=None):
        super().__init__(None)  # base class doesn't get a factory
        self.f_of_x = f_of_x  # save f(x)

    def __missing__(self, key):  # called when a default needed
        ret = self.f_of_x(key)  # calculate default value
        self[key] = ret  # and install it in the dict
        return ret








def binary_integer_format(integer, number_of_bits):
    return "{0:b}".format(integer).zfill(number_of_bits)


def get_enumerated_rev_map_from_indices(indices):
    enumerated_dict = dict(enumerate(indices))
    rev_map = {}
    for k, v in enumerated_dict.items():
        rev_map[v]=k
    return rev_map

def get_enumerated_rev_map_from_dict(enumerated_dict):
    rev_map = {}
    for k, v in enumerated_dict.items():
        rev_map[v]=k
    return rev_map


def get_qubit_indices_from_string(qubits_string,
                                  with_q=False):

    """Return list of qubit indices from the string of the form "q0q1q22q31"
    :param qubits_string (string): string which has the form of "q" followed by qubit index
    :param (optional) with_q (Boolean): specify whether returned indices should be in form of string with letter

    :return: list of qubit indices:

    depending on value of parameter "with_q" the mapping will be one of the following:

    if with_q:
        'q1q5q13' -> ['q1','q5','q13']
    else:
        'q1q5q13' -> [1,5,13]

    """

    numbers = re.findall(r'\d+', qubits_string)

    if with_q:
        qubits = ['q'+s for s in numbers]
    else:
        qubits = [int(s) for s in numbers]

    return qubits


def round_matrix(m_a, decimal):
    typ = type(m_a[0, 0])

    n = m_a.shape[0]
    m = np.size(m_a, axis=1)

    m_b = np.zeros((n, m), dtype=typ)

    for i in range(0, n):
        for j in range(0, m):
            x = round(np.real(m_a[i, j]), decimal)

            if typ == complex or typ == np.complex128:
                y = round(np.imag(m_a[i, j]), decimal)
            else:
                y = 0

            # TODO TR: Define intention or provide a formula (or some kind of source) or describe intention.
            # it's so complicated because I don't like the look of "minus 0"

            if abs(x) != 0 and abs(y) != 0:
                if typ == complex or typ == np.complex128:
                    m_b[i, j] = x + 1j * y
                else:
                    m_b[i, j] = x
            elif abs(x) == 0 and abs(y) == 0:
                m_b[i, j] = 0
            elif abs(x) == 0 and abs(y) != 0:
                if typ == complex or typ == np.complex128:
                    m_b[i, j] = 1j * y
                else:
                    m_b[i, j] = 0
            elif abs(x) != 0 and abs(y) == 0:
                m_b[i, j] = x

    return m_b


# TODO TR: This function should be renamed.
def sandwich(m_a, m_u):
    u_dg = np.matrix.getH(m_u)
    m_x = np.matmul(m_u, np.matmul(m_a, u_dg))
    return m_x


# ===========================================================================
# Functions that checks if matrix is zero. Function returns parameter, that
# for zero matrix will be equal to 0, and for non-zero matrix - equal to 1.
# Method: searching through all elements and thresholding.
# ===========================================================================
def zero_check(m_a, eps=epsilon):
    size = list(m_a.shape)
    m_b = copy.deepcopy(m_a)
    if eps >= 1:
        decimal = eps
        eps = 10 ** (-decimal)

    if len(size) == 1:
        m_b = m_a.reshape(size[0], 1)
        size.append(1)

    for i in range(0, size[0]):
        for j in range(0, size[1]):
            check_zero = np.abs(m_b[i, j])
            if check_zero > eps:
                return False
    return True


# TODO TR: Check whether variables has been properly renamed.
def spectral_decomposition(m_a):
    eigen_values, eigen_vectors = np.linalg.eig(m_a)

    d = m_a.shape[0]
    projectors = [calculate_outer_product(np.array(eigen_vectors[:, i]).reshape(d, 1)) for i in range(d)]

    return eigen_values, projectors


def calculate_outer_product(ket):
    return ket @ np.matrix.getH(ket)


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


# TODO TR: Comment this method, and intention behind it.
def thresh(m_a, decimal=7):
    m_b = np.array(copy.deepcopy(m_a))
    with np.nditer(m_b, op_flags=['readwrite']) as it:
        for x in it:
            x[...] = np.round(x, decimal)

    return m_b





def lists_intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))


def lists_difference(lst1, lst2):
    return list(set(lst1) - set(lst2))


def lists_sum(lst1, lst2):
    return list(set(lst1).union(set(lst2)))

def lists_sum_multi(lists):
    return set().union(*lists)

def lists_intersection_multi(lists):
    l0 = lists[0]
    l1 = lists[1]

    int_list = lists_intersection(l0,l1)
    for l in lists[2:]:
        int_list = lists_intersection(int_list,l)
    return int_list

def check_if_there_are_common_elements(lists):

    for i in range(len(lists)):
        for j in range(i+1, len(lists)):
            if len(lists_intersection(lists[i],lists[j]))!=0:
                return True

    return False


def find_significant_digit(x):
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

def cool_print(a,b='',color=Fore.CYAN):
    #a is printed with color
    #b is printed without color
    if b=='':
        print(color+Style.BRIGHT+str(a)+Style.RESET_ALL)
    else:
        print(color+Style.BRIGHT+str(a)+Style.RESET_ALL,repr(b))


def bit_strings(n, rev=False):
    """Generate outcome bitstrings for n-qubits.

    Args:
        n (int): the number of qubits.

    Returns:
        list: arrray_to_print list of bitstrings ordered as follows:
        Example: n=2 returns ['00', '01', '10', '11'].
"""
    if (rev == True):
        return [(bin(j)[2:].zfill(n))[::-1] for j in list(range(2 ** n))]
    else:
        return [(bin(j)[2:].zfill(n)) for j in list(range(2 ** n))]



def register_names_qubits(qs, qrs, rev=False):
    if qrs == 0:
        return ['']

    if (qrs == 1):
        return ['0', '1']

    all_names = bit_strings(qrs, rev)
    not_used = []

    for j in list(range(qrs)):
        if j not in qs:
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
from povms_qi import ancillary_functions


def zeros_to_dots(A, decimal):
    m = A.shape[0]
    n = A.shape[1]

    B = np.zeros((m, n), dtype=dict)

    for i in range(m):
        for j in range(n):
            el = A[i, j]
            if (abs(np.round(el, decimal)) >= 0):
                B[i, j] = el
            else:
                B[i, j] = '.'

    return B


def print_array_nicely(arrray_to_print,
                       rounding_decimal=3):
    import pandas as pd
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
        print(pd.DataFrame(np.array(np.round(arrray_to_print, rounding_decimal))))

