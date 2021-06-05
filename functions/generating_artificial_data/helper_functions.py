"""
@authors: Filip Maciejewski, Oskar Słowik, Tomek Rybotycki
@contact: filip.b.maciejewski@gmail.com
"""
import numpy as np
from typing import List
import copy


def get_1q_stochastic_matrix(p10: float, p01: float) -> np.ndarray:
    """ Get single-qubit stochastic map from error probabilities.

    :param p10: probability of getting outcome "1" if input was |0>
    :param p01: probability of getting outcome "0" if input was |1>

    :return: single-qubit stochastic matrix
    """

    return np.array([[1 - p01, p10], [p01, 1 - p10]])


def get_random_stochastic_matrix_1q(p10_range: list,
                                    p01_range: list,
                                    distribution_name='uniform') -> np.ndarray:
    """ Get random single-qubit stochastic map with error probabilities chosen randomly from provided
    ranges

    :param p10_range: [min,max] value of probability of getting outcome "1" if input was |0>
    :param p01_range: [min,max] value of probability of getting outcome "0" if input was |1>
    :param distribution_name: stochasticity_type of distribution
    possible options:
    - "uniform"

    #todo FBM: add Gaussian types

    :return: single-qubit stochastic matrix
    """

    if distribution_name == 'uniform':
        p10 = np.random.uniform(*p10_range)
        p01 = np.random.uniform(*p01_range)
    else:
        raise ValueError('Wrong distribution name')

    return get_1q_stochastic_matrix(p10, p01)


def add_physical_perturbation_1q(vector: np.ndarray,
                                 perturbation_range):
    changed_p10 = False
    perturbed_vector = copy.deepcopy(vector)
    while not changed_p10:
        # try to add perturbation to probability vector that is still physical
        magnitude_p10, sign_p10 = np.random.uniform(*perturbation_range), np.random.choice([-1, 1])
        p0 = vector[0]
        p1 = 1 - p0

        if 0 <= (p0 + sign_p10 * magnitude_p10) <= 1 and 0 <= (p1 - sign_p10 * magnitude_p10) <= 1:
            perturbed_vector[0] = p0 + sign_p10 * magnitude_p10
            perturbed_vector[1] = p1 - sign_p10 * magnitude_p10
            changed_p10 = True
        elif 0 <= (p0 - sign_p10 * magnitude_p10) <= 1 and 0 <= (p1 + sign_p10 * magnitude_p10) <= 1:
            perturbed_vector[0] = p0 - sign_p10 * magnitude_p10
            perturbed_vector[1] = p1 + sign_p10 * magnitude_p10
            changed_p10 = True
        else:
            pass

    return perturbed_vector


def perturb_stochastic_matrix_1q(stochastic_matrix: np.ndarray,
                                 deviation_p10_range: List[float],
                                 deviation_p01_range: List[float]) -> np.ndarray:
    perturbed_matrix = copy.deepcopy(stochastic_matrix)

    first_column_perturbed = add_physical_perturbation_1q(perturbed_matrix[:, 0], deviation_p10_range)
    second_column_perturbed = add_physical_perturbation_1q(perturbed_matrix[:, 1], deviation_p01_range)

    perturbed_matrix[:, 0] = first_column_perturbed[:]
    perturbed_matrix[:, 1] = second_column_perturbed[:]

    return perturbed_matrix
