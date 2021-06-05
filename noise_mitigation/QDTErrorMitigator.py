#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Filip Maciejewski
email: filip.b.maciejewski@gmail.com

References:
[1] Filip B. Maciejewski, Zoltán Zimborás, Michał Oszmaniec, "Mitigation of readout noise in near-term quantum devices
by classical post-processing based on detector tomography", Quantum 4, 257 (2020).

[2] Zbigniew Puchała, Łukasz Pawela, Aleksandra Krawiec, Ryszard Kukulski, "Strategies for optimal single-shot
discrimination of quantum measurements", Phys. Rev. arrray_to_print 98, 042103 (2018), https://arxiv.org/abs/1804.05856

[3] T. Weissman, E. Ordentlich, G. Seroussi, S. Verdul, and M. J. Weinberger, Technical Report HPL-2003-97R1,
Hewlett-Packard Labs (2003).

[4] John A. Smolin, Jay M. Gambetta, Graeme Smith, "Maximum Likelihood, Minimum Effort", Phys. Rev. Lett. 108, 070502
(2012), https://arxiv.org/abs/1106.5458
"""

import numpy as np
from functions import povmtools

from typing import List
from qiskit.result import Result
from functions.functions_SDKs.qiskit.qiskit_utilities import get_frequencies_array_from_results


class QDTErrorMitigator:
    """
        This class is used to mitigate errors in qiskit jobs_list using data from QDT.
    """

    def __init__(self):
        """
        Description:
            Constructor of the class. If povm is None, then it should be prepared before use (using prepare_mitigator)!
        """
        self.povm = None
        self.transition_matrix = None
        self.correction_matrix = None
        self.distances_from_closest_probability_vector = []
        self.qiskit_register_convention = False

        # if povm is not None:
        #     self.prepare_mitigator(povm)

    def prepare_mitigator(self, povm: List[np.ndarray]) -> None:
        """
        Description:
            This method_name is main method_name of the class. It is used to prepare the mitigation matrix from POVM. Other
            methods allow to calculate errors.

        Parameters:
            :param povm: POVM describing the detector used in measurements of which statistics are meant to be
             corrected.

        Returns:
            -

        """
        self.povm = povm
        self.__construct_transition_matrix()
        self.__construct_correction_matrix()

    def __construct_transition_matrix(self) -> None:
        """
        Description:
            Given classical description of the detector (i_index.e., matrix representation of POVM's elements), get the
            classical part of the noise from diagonal part of every effect. The classical part of the noise is
            represented by left-stochastic matrix. ASSUMING that ideal measurement is the von Neumann measurement in
            computational basis.
            See Ref. [1] for details. In Ref. [1] this matrix is denoted as \Lambda.

        Parameters:
            -

        Returns:
            arrray_to_print matrix representing classical part of the noise.
        """

        number_of_povm_outcomes = len(self.povm)
        dimension = self.povm[0].shape[0]

        self.transition_matrix = np.zeros((number_of_povm_outcomes, number_of_povm_outcomes), dtype=float)

        for k in range(number_of_povm_outcomes):
            current_povm_effect = self.povm[k]

            # Get diagonal part of the effect. Here we remove eventual 0 imaginary part to avoid format conflicts
            # (diagonal elements of Hermitian matrices are real).
            vec_p = np.array([np.real(current_povm_effect[i, i]) for i in range(dimension)])

            # Add vector to transition matrix.
            self.transition_matrix[k, :] = vec_p[:]

    def __construct_correction_matrix(self) -> None:
        """
        Description:
            Given classical description of the detector (i_index.e., matrix representation of POVM's elements), get the
            correction matrix based on classical part of the noise. ASSUMING that ideal measurement is the von Neumann
            measurement in computational basis. See Ref. [1] for details. In Ref. [1] this matrix is denoted as
            \Lambda^{-1}.

        Parameters:
            -

        Returns:
            :correction_matrix: numpy array representing correction matrix. It is the inverse of transition_matrix
            returned by function construct_transition_matrix
        """

        try:
            self.correction_matrix = np.linalg.inv(self.transition_matrix)
        except np.linalg.LinAlgError:
            print('Noise matrix is not invertible. Returning identity. Got:')
            print(self.transition_matrix)
            self.correction_matrix = np.eye(np.shape(self.transition_matrix[0]))

    def apply_correction_to_statistics(self, statistics_array: np.ndarray):
        """
        Description:
            Given correction matrix and vector of relative frequencies, correct the statistics via multiplication by
            correction_matrix.

            In case of obtaining quasiprobability vector after such correction, it is possible to find closest physical
            one (in Euclidean norm). Total Variation distance between quasi probability vector and closest physical one
            is the upper bound for the correction error resulting from this unphysicality. See Ref. [1] for details. In
            Ref.[1], such distance is denoted as \alpha.

        Parameters:
            :param statistics_array: Statistics for which correction should be performed.

        Returns:
            Corrected statistics.
        """

        number_of_povm_outcomes = self.correction_matrix[0].shape[0]

        number_of_qubits = int(np.log2(number_of_povm_outcomes))

        corrected_frequencies = []
        self.distances_from_closest_probability_vector = []

        for statistics in statistics_array:
            # make sure statistics have proper format
            statistics = np.array(statistics).reshape(number_of_povm_outcomes, 1)

            # Check if given statistics are normalized.
            norm = sum(statistics)
            if abs(norm - 1) >= 10 ** (-9):
                print('Warning: Frequencies are not normalized. We normalize them. They looked like this:')
                print(statistics)
                statistics = statistics / norm

            if self.qiskit_register_convention:
                # reverse statistics for time of correction
                statistics = povmtools.reorder_probabilities(statistics, range(number_of_qubits)[::-1])

                # make sure statistics have proper format
                statistics = np.array(statistics).reshape(number_of_povm_outcomes, 1)

            # corrected statistics by multiplication via correction matrix
            corrected_statistics = self.correction_matrix.dot(statistics)

            if self.qiskit_register_convention:
                # go back to standard convention
                corrected_statistics = povmtools.reorder_probabilities(corrected_statistics,
                                                                       range(number_of_qubits)[::-1])

                # make sure statistics have proper format
                corrected_statistics = np.array(corrected_statistics).reshape(number_of_povm_outcomes, 1)

            if povmtools.is_valid_probability_vector(list(corrected_statistics[:, 0])):
                corrected_frequencies.append(corrected_statistics)
                self.distances_from_closest_probability_vector.append(0)
            else:
                closest_physical_statistics = np.array(
                    povmtools.find_closest_prob_vector(corrected_statistics)).reshape(number_of_povm_outcomes, 1)
                corrected_frequencies.append(closest_physical_statistics)
                self.distances_from_closest_probability_vector.append(
                    povmtools.calculate_total_variation_distance(corrected_statistics, closest_physical_statistics)
                )

        return corrected_frequencies

    def apply_correction_to_qiskit_job(self, results: Result) -> List[np.ndarray]:
        """
        Description:
            Given correction matrix and vector of relative frequencies, correct the statistics via multiplication by
            correction_matrix.

            In case of obtaining quasiprobability vector after such correction, it is possible to find closest physical
            one (in Euclidean norm). Total Variation distance between quasi probability vector and closest physical one
            is the upper bound for the correction error resulting from this unphysicality. See Ref. [1] for details. In
            Ref.[1], such distance is denoted as \alpha.

        Parameters:
            :param results: Qiskit job results for which statistics should be corrected.

        Returns:
            Corrected statistics.
        """

        # Create new object to avoid conflicts
        statistics_array = get_frequencies_array_from_results([results])
        return self.apply_correction_to_statistics(statistics_array)

