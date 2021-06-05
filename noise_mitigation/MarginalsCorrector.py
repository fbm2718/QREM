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

[1] B. Nachman, M. Urbanek, W. arrray_to_print. de Jong, C. W. Bauer,
 "Unfolding quantum computer readout noise",
 npj Quantum Inf 6, 84 (2020)
"""

import numpy as np
from typing import Optional, Dict, List
from base_classes.marginals_analyzer_base import MarginalsAnalyzerBase
from functions import povmtools as pt, ancillary_functions as anf


class MarginalsCorrector(MarginalsAnalyzerBase):
    """
    This is the main class that uses correction data to reduce noise on the level of marginals_dictionary.
    Main functionalities are to correct marginal distributions of some experiments.

    NOTE: please see parent class MarginalsAnalyzerBase for conventions of experimental results storage

    """

    def __init__(self,
                 experimental_results_dictionary: Dict[str, Dict[str, int]],
                 bitstrings_right_to_left: bool,
                 correction_data_dictionary: dict,
                 marginals_dictionary: Optional[Dict[str, Dict[str, np.ndarray]]] = None
                 ) -> None:
        """
        :param experimental_results_dictionary: dictionary of results of experiments we wish to correct
        (see class' description for conventions)

        :param bitstrings_right_to_left: specify whether bitstrings
                                    should be read from right to left (when interpreting qubit labels)
        :param correction_data_dictionary: dictionary that contains information needed for noise
                                           mitigation on marginal probability distributions.


        :param marginals_dictionary: in case we pre-computed some marginal distributions
                                     (see class' description for conventions)

        """

        super().__init__(experimental_results_dictionary,
                         bitstrings_right_to_left,
                         marginals_dictionary
                         )

        self._noise_matrices = correction_data_dictionary['noise_matrices']

        if 'correction_matrices' in correction_data_dictionary.keys():
            self._correction_matrices = correction_data_dictionary['correction_matrices']
        else:
            anf.cool_print('No correction matrices provided!','','red')
        self._correction_indices = correction_data_dictionary['correction_indices']

        self._corrected_marginals = {}

    @property
    def correction_indices(self) -> Dict[str, str]:
        return self._correction_indices

    @correction_indices.setter
    def correction_indices(self, correction_indices) -> None:
        self._correction_indices = correction_indices

    @property
    def corrected_marginals(self) -> Dict[str, Dict[str, np.ndarray]]:
        return self._corrected_marginals

    def get_specific_marginals_from_marginals_dictionary(self,
                                                         keys_of_interest: List[str],
                                                         corrected=False) -> \
            Dict[str, Dict[str, np.ndarray]]:
        """From dictionary of marginals_dictionary take only those which are in "marginals_labels_hamiltonian".
        Furthermore, for all keys, calculate also two-qubit and single-qubit marginals_dictionary for qubits
        inside those marginals_dictionary.

        :param keys_of_interest: list of strings representing qubit indices, e.g., 'q1q3q15'

        :return: marginals_of_interest : dictionary with marginal distributions for marginals_labels_hamiltonian
        """
        if corrected:
            marginals_dictionary = self._corrected_marginals
        else:
            marginals_dictionary = self._marginals_dictionary

        marginals_of_interest = {}

        for key in marginals_dictionary.keys():
            distribution_now = marginals_dictionary[key]
            if key in keys_of_interest:
                marginals_of_interest[key] = distribution_now

            qubits_here = self.get_qubit_indices_from_string(key)
            enumerated_qubits = dict(enumerate(qubits_here))
            rev_map = {}
            for kkk, vvv in enumerated_qubits.items():
                rev_map[vvv] = kkk

            for qi in qubits_here:
                if 'q%s' % qi in keys_of_interest:
                    marginals_of_interest['q%s' % qi] = \
                        self.get_marginal_from_probability_distribution(
                            distribution_now, [rev_map[qi]])

            for qi in qubits_here:
                for qj in qubits_here:
                    if 'q%sq%s' % (qi, qj) in keys_of_interest:
                        marginals_of_interest[
                            'q%sq%s' % (qi, qj)] = self.get_marginal_from_probability_distribution(
                            distribution_now, sorted([rev_map[qi], rev_map[qj]]))

        return marginals_of_interest

    @staticmethod
    def correct_distribution_T_matrix(distribution: np.ndarray,
                                      correction_matrix: np.ndarray,
                                      ensure_physicality=True):
        """Correct probability distribution using multiplication via inverse of noise matrix.
        See Refs. [0], [0.5].

        :param distribution: noisy distribution
        :param correction_matrix: correction matrix (inverse of noise matrix)
        :param ensure_physicality: if True, then after correcting marginal distribution it ensures that
                                    resulting vector has elements from [0,1]

        :return: array representing corrected distribution

        #TODO FBM: add option to return mitigation errors
        """

        # Consistent formatting
        if isinstance(distribution, list):
            d = len(distribution)
        elif isinstance(distribution, np.ndarray):
            d = distribution.shape[0]
        else:
            raise TypeError("Wrong distribution format")

        distribution = np.array(distribution).reshape(d, 1)

        # correct distribution using inverse of noise matrix
        quasi_distribution = correction_matrix.dot(distribution)

        # do we want to ensure resulting distribution is physical?
        if ensure_physicality:
            # if yes, check if it is physical
            if pt.is_valid_probability_vector(quasi_distribution):
                # if it is physical, no further action is required
                return quasi_distribution
            else:
                # if it is not physical, find closest physical one
                return np.array(pt.find_closest_prob_vector(quasi_distribution)).reshape(d, 1)

        else:
            # if we don't care about physicality, we don't do anything and just return vector
            return quasi_distribution

    @staticmethod
    def correct_distribution_IBU(estimated_distribution: np.ndarray,
                                 noise_matrix: np.ndarray,
                                 iterations_number: Optional[int] = 10,
                                 prior: Optional[np.ndarray] = None):
        """Correct probability distribution using Iterative Bayesian Unfolding (IBU)
        See Ref. [1]

        :param estimated_distribution: noisy distribution (to be corrected)
        :param noise_matrix: noise matrix
        :param iterations_number: number of iterations in unfolding
        :param prior: ansatz for ideal distribution, default is uniform

        :return: array representing corrected distribution
        """

        # Consistent formatting
        if isinstance(estimated_distribution, list):
            d = len(estimated_distribution)
        elif isinstance(estimated_distribution, np.ndarray):
            d = estimated_distribution.shape[0]
        else:
            raise TypeError("Wrong distribution format")

        # default iterations number
        if iterations_number is None:
            iterations_number = 25

        # If no prior is provided, we take uniform distribution
        if prior is None or prior == 'uniform':
            prior = np.full((d, 1), 1 / d, dtype=float)

        # initialize distribution for recursive loop
        distribution_previous = prior
        # go over iterations
        for iteration_index in range(iterations_number):
            distribution_new = np.zeros((d, 1), dtype=float)

            # go over measurement outcomes
            for outcome_index in range(d):
                # calculating IBU estimate
                distribution_new[outcome_index] = sum(
                    [noise_matrix[j, outcome_index] * distribution_previous[outcome_index] *
                     estimated_distribution[j] / sum(
                        [noise_matrix[j, k] * distribution_previous[k] for k in range(d)])
                     for j in
                     range(d)])
            # update distribution for recursion
            distribution_previous = distribution_new

        return distribution_previous

    # @staticmethod
    def correct_distribution_hybrid_T_IBU(self,
                                          estimated_distribution,
                                          noise_matrix,
                                          correction_matrix: Optional[np.ndarray] = None,
                                          unphysicality_threshold: Optional[float] = 0.0,
                                          iterations_number: Optional[int] = None,
                                          prior: Optional[np.ndarray] = None):
        """
        Correct distribution using method_name that is hybrid of T_matrix correction (see Refs. [0,0.5])
        and Iterative Bayesian Unfolding (IBU) (see Ref. [1]).

        Algorithm goes like this:
        - Correct distribution using inverse of noise matrix.
        - If distribution is physical (i_index.e., elements are from (0,1)), return it.
        - Otherwise, perform IBU.
        - [Optional] if parameter "unphysicality_threshold" is provided,
           then it projects the result of "T_matrix" correction onto probability simplex
           and if Total Variation Distance between this projected vector
           and original unphysical one is belowthis cluster_threshold, then it returns the projection.
           Otherwise, it performs IBU.

        :param estimated_distribution: noisy distribution (to be corrected)
        :param noise_matrix: noise matrix
        :param correction_matrix: correction matrix (inverse of noise matrix)
                                 if not provided, it is calculated from noise_matrix
        :param unphysicality_threshold: cluster_threshold to decide whether unphysical "T_matrix"
                                        correction is acceptable. See description of the function
        :param iterations_number: number of iterations in IBU
        :param prior: ansatz for ideal distribution in IBU, default is uniform

        """

        if isinstance(estimated_distribution, list):
            d = len(estimated_distribution)
        elif isinstance(estimated_distribution, np.ndarray):
            d = estimated_distribution.shape[0]
        else:
            raise TypeError("Wrong distribution format")

        if correction_matrix is None:
            correction_matrix = np.linalg.inv(noise_matrix)

        distribution = np.array(estimated_distribution).reshape(d, 1)

        quasi_distribution = correction_matrix.dot(distribution)

        is_physical = pt.is_valid_probability_vector(quasi_distribution)

        if is_physical:
            return quasi_distribution
        else:
            if unphysicality_threshold == 0:
                return self.correct_distribution_IBU(distribution,
                                                     noise_matrix,
                                                     iterations_number,
                                                     prior)
            else:
                closest_physical = pt.find_closest_prob_vector(quasi_distribution)
                distance_now = 1 / 2 * np.linalg.norm(quasi_distribution - closest_physical, ord=1)
                if distance_now <= unphysicality_threshold:
                    return closest_physical

                else:
                    return self.correct_distribution_IBU(distribution,
                                                         noise_matrix,
                                                         iterations_number,
                                                         prior)

    def correct_marginals(self,
                          marginals_dictionary: Optional[Dict[str, np.ndarray]] = None,
                          method='T_matrix',
                          method_kwargs=None) -> Dict[str, Dict[str, np.ndarray]]:

        """Return dictionary of corrected marignal distributions
        :param marginals_dictionary: dictionary of (noisy) marginal distributions
        :param method: method_name to be used for correction of marginal probability distributions

        possible values:
        - 'T_matrix' - uses inverse of noise matrix as correction (see Refs. [0,0.5])
        - 'IBU' - uses Iterative Bayesian Unfolding (see Ref. [1])
        - 'hybrid_T_IBU' - uses hybrid method_name between 'T_matrix' and 'IBU',
                          see description of self.correct_distribution_hybrid_T_IBU

        :param method_kwargs:  keyword arguments passed to function using chosen method_name.
                               See description of specific functions.

        :return: corrected_marginals : dictionary of corrected marginal distributions
        """

        # TODO FBM: make it consistent with conventions used in parent class
        if marginals_dictionary is None:
            marginals_dictionary = self._marginals_dictionary

        corrected_marginals = {}

        for key in marginals_dictionary.keys():

            noisy_marginal_now = marginals_dictionary[key]

            if method == 'T_matrix':
                if method_kwargs is None:
                    method_kwargs = {'ensure_physicality': True}
                corrected_marginal_now = self.correct_distribution_T_matrix(noisy_marginal_now,
                                                                            self._correction_matrices[
                                                                                key],
                                                                            **method_kwargs)
            elif method == 'IBU':
                if method_kwargs is None:
                    method_kwargs = {'iterations_number': None,
                                     'prior': None}

                corrected_marginal_now = self.correct_distribution_IBU(noisy_marginal_now,
                                                                       self._noise_matrices[
                                                                           key],
                                                                       **method_kwargs)
            elif method == 'hybrid_T_IBU':
                if method_kwargs is None:
                    method_kwargs = {'unphysicality_threshold': 0.0,
                                     'iterations_number': None,
                                     'prior': None}

                corrected_marginal_now = self.correct_distribution_hybrid_T_IBU(noisy_marginal_now,
                                                                                self._noise_matrices[
                                                                                    key],
                                                                                self._correction_matrices[
                                                                                    key],
                                                                                **method_kwargs)
            else:
                raise ValueError('Wrong method_name name')

            corrected_marginals[key] = corrected_marginal_now

        self._corrected_marginals = corrected_marginals
        return corrected_marginals
