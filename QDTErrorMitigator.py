#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Filip Maciejewski
email: filip.b.maciejewski@gmail.com

References:
[1] Filip B. Maciejewski, Zoltán Zimborás, Michał Oszmaniec, "Mitigation of readout noise in near-term quantum devices
by classical post-processing based on detector tomography", arxiv preprint, https://arxiv.org/abs/1907.08518 (2019)

[2] Zbigniew Puchała, Łukasz Pawela, Aleksandra Krawiec, Ryszard Kukulski, "Strategies for optimal single-shot
discrimination of quantum measurements", Phys. Rev. A 98, 042103 (2018), https://arxiv.org/abs/1804.05856

[3] T. Weissman, E. Ordentlich, G. Seroussi, S. Verdul, and M. J. Weinberger, Technical Report HPL-2003-97R1,
Hewlett-Packard Labs (2003).

[4] John A. Smolin, Jay M. Gambetta, Graeme Smith, "Maximum Likelihood, Minimum Effort", Phys. Rev. Lett. 108, 070502
(2012), https://arxiv.org/abs/1106.5458
"""

import numpy as np
import povmtools
from typing import List
from qiskit.result import Result


class QDTErrorMitigator:
    """
        This class is used to mitigate errors in qiskit jobs using data from QDT.
    """

    def __init__(self):
        """
        Description:
            Constructor of the class. 
        """
        self._povm = None
        self._transition_matrix = None
        self._correction_matrix = None
        self._coherent_error_bound = None
        self._statistical_errors_bounds = []  # Possibly there are several frequencies.
        self.statistical_error_mistake_probability = 0.01  # Can be accessed and changed at will.
        self._qubit_indices_lists = []
        self._distances_from_closest_probability_vector = []

    def prepare_mitigator(self, povm: List[np.ndarray], qubit_indices_lists: List[List[int]]):
        """
        Description:
            This method is main method of the class. It is used to prepare the mitigation matrix from POVM. Other methods allow to calculate errors. 

        Parameters:
            :param povm: POVM describing the detector.
            :param qubit_indices_lists: Indices for which QDT was performed. Elements of this list should correspond to
            elements in povms_list. They will be used in statistics correction.

        Returns:
            -

        """
        self._qubit_indices_lists = qubit_indices_lists
        self._povm = povm
        self.__construct_transition_matrix()
        self.__construct_correction_matrix()
        
        

    def __construct_transition_matrix(self) -> None:
        """
        Description:
            Given classical description of the detector (i.e., matrix representation of POVM's elements), get the
            classical part of the noise from diagonal part of every effect. The classical part of the noise is
            represented by left-stochastic matrix. ASSUMING that ideal measurement is the von Neumann measurement in
            computational basis.
            See Ref. [1] for details. In Ref. [1] this matrix is denoted as \Lambda.

        Parameters:
            -

        Returns:
            A matrix representing classical part of the noise.
        """

        number_of_povm_outcomes = len(self._povm)
        dimension = self._povm[0].shape[0]

        self._transition_matrix = np.zeros((number_of_povm_outcomes, number_of_povm_outcomes), dtype=float)

        for k in range(number_of_povm_outcomes):
            current_povm_effect = self._povm[k]

            # Get diagonal part of the effect. Here we remove eventual 0 imaginary part to avoid format conflicts
            # (diagonal elements of Hermitian matrices are real).
            vec_p = np.array([np.real(current_povm_effect[i, i]) for i in range(dimension)])

            # Add vector to transition matrix.
            #TODO FBM: changed here because there was mistake
            self._transition_matrix[k,:] = vec_p[:]

    def __construct_correction_matrix(self) -> None:
        """
        Description:
            Given classical description of the detector (i.e., matrix representation of POVM's elements), get the correction
            matrix based on classical part of the noise. ASSUMING that ideal measurement is the von Neumann measurement in
            computational basis. See Ref. [1] for details. In Ref. [1] this matrix is denoted as \Lambda^{-1}.

        Parameters:
            -

        Returns:
            :correction_matrix: numpy array representing correction matrix. It is the inverse of transition_matrix returned by function construct_transition_matrix
        """

        try:
            self._correction_matrix = np.linalg.inv(self._transition_matrix)
        except np.linalg.LinAlgError:
            print('Noise matrix is not invertible. Returning identity. Got:')
            print(self._transition_matrix)
            self._correction_matrix = np.eye(np.shape(self._transition_matrix[0]))

    def apply_correction_to_qiskit_job(self, results: Result, qiskit_register_convention = True) -> List[np.ndarray]:
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

        number_of_povm_outcomes = self._correction_matrix[0].shape[0]
        
        number_of_qubits = int(np.log2(number_of_povm_outcomes))
        
        # create new object to avoid conflicts
        statistics_array = self.__get_frequencies_array_from_results(results)
        corrected_frequencies = []
        self._distances_from_closest_probability_vector = []
        


        for statistics in statistics_array:
            # make sure statistics have proper format
            statistics = np.array(statistics).reshape(number_of_povm_outcomes, 1)
            
            
            # check norm
            norm = sum(statistics)
            if abs(norm - 1) >= 10 ** (-9):
                print('Warning: Frequencies are not normalized. We normalize them. They looked like this:')
                print(statistics)
                statistics = statistics / norm

           
            if qiskit_register_convention:
                #reverse ordering of statistics for time of correction
                statistics=povmtools.reorder_probabilities(statistics,range(number_of_qubits)[::-1])


                # make sure statistics have proper format
                statistics = np.array(statistics).reshape(number_of_povm_outcomes, 1)

            # corrected statistics by multiplication via correction matrix
            corrected_statistics = self._correction_matrix.dot(statistics)
            
            
            if qiskit_register_convention:
                #go back to standard convention 
                corrected_statistics=povmtools.reorder_probabilities(corrected_statistics,range(number_of_qubits)[::-1])
                
                # make sure statistics have proper format
                corrected_statistics=np.array(corrected_statistics).reshape(number_of_povm_outcomes,1)


            if povmtools.is_valid_probability_vector(list(corrected_statistics[:, 0])):
                corrected_frequencies.append(corrected_statistics)
                self._distances_from_closest_probability_vector.append(0)
            else:
                closest_physical_statistics = np.array(povmtools.find_closest_prob_vector(corrected_statistics)).reshape(number_of_povm_outcomes,1)
                corrected_frequencies.append(closest_physical_statistics)
                self._distances_from_closest_probability_vector.append(
                    povmtools.calculate_total_variation_distance(corrected_statistics, closest_physical_statistics)
                )
        # print()
        # print(self._distances_from_closest_probability_vector)
                
                
                
        return corrected_frequencies

    @staticmethod
    # TODO TR: This is duplicate code. Consider placing this in another file.
    def __get_frequencies_array_from_results(results: Result) -> np.ndarray:
        """
        Description:
            Creates an array of frequencies from given qiskit job results. This method is is working with
            qiskit 0.16. The shape of the array is

                c x 2 ** q,

            where c denotes circuits number and q denotes number of qubits.

        Parameters:
            :param results: qiskit jobs results

        Returns:
            ndarray with shape=0 if there were no circuits in the job, or with shape c x 2 ** q
            containing frequencies data for each possible state.

        Notes:
            Possible states are numbered increasingly from |00000 ... 0>, |10000 ... 0> up to |1111 ... 1>.
        """

        circuits_number = len(results.results)

        if circuits_number == 0:
            return np.ndarray(shape=0)

        
        # states_len = len(next(iter(results.get_counts(0).keys())))
        # The length of a state describes how many qubits were used during experiment. Assuming that all results have are on the same number of qubits.
        states_len=np.max([len(list(key)) for key in results.get_counts(0).keys()])
        

        possible_states = ["{0:b}".format(i) for i in range(2 ** states_len)]

        for i in range(len(possible_states)):
            while len(possible_states[i]) < states_len:
                possible_states[i] = '0' + possible_states[i]

        frequencies_array = np.ndarray(shape=(circuits_number, len(possible_states)))


        def fix_fromat_counts_leading_zeros(cnts):
            #TODO FBM: I encountered some problems with test objects due to HEXADECIMAL formatting in qiskit... Here I fix it. We may probably remove it later
            keys=list(cnts.keys())
            
            string_lengths=np.unique([len(list(key)) for key in keys])
            
            new_cnts={}
            if len(string_lengths)!=1:
                proper_length=np.max(string_lengths)

                for key in keys:
                    if len(list(key))!=proper_length:
                        new_count=bin(int(key,2))[2:].zfill(proper_length)
                        new_cnts[new_count]=cnts[key]
                    else:
                        new_cnts[key]=cnts[key]

                
                return new_cnts
            else:
                print('good format')
                return cnts
            

        for i in range(circuits_number):
            counts = results.get_counts(i)
            counts = fix_fromat_counts_leading_zeros(counts)
          
                
            shots_number = results.results[i].shots
            for j in range(len(possible_states)):
                if possible_states[j] in counts.keys():
                    frequencies_array[i][j] = counts[possible_states[j]] / shots_number
                else:
                    frequencies_array[i][j] = 0
                
                

        return frequencies_array

    @staticmethod
    # TODO TR: This is similar to frequencies array creation. Should be refactored.
    def __get_counts_array_from_results(results: Result) -> np.ndarray:
        """
        Description:
            Creates an array of frequencies from given qiskit job results. This method is working with
            qiskit 0.16. The shape of the array is

                c x 2 ** q,

            where c denotes circuits number and q denotes number of qubits.

        Parameters:
            :param results: qiskit jobs results

        Returns:
            ndarray with shape=0 if there were no circuits in the job, or with shape c x 2 ** q
            containing frequencies data for each possible state.

        Notes:
            Possible states are numbered increasingly from |00000 ... 0>, |10000 ... 0> up to |1111 ... 1>.
        """

        circuits_number = len(results.results)

        if circuits_number == 0:
            return np.ndarray(shape=0)

        # The length of a state describes how many qubits were used during experiment.
        states_len = len(next(iter(results.get_counts(0).keys())))

        possible_states = ["{0:b}".format(i) for i in range(2 ** states_len)]

        for i in range(len(possible_states)):
            while len(possible_states[i]) < states_len:
                possible_states[i] = '0' + possible_states[i]

        frequencies_array = np.ndarray(shape=(circuits_number, len(possible_states)))

        for i in range(circuits_number):
            counts = results.get_counts(i)
            for j in range(len(possible_states)):
                if possible_states[j] in counts.keys():
                    frequencies_array[i][j] = counts[possible_states[j]]
                else:
                    frequencies_array[i][j] = 0

        return frequencies_array

    def __calculate_coherent_error_bound(self) -> None:
        """
        Description:
            Get distance between diagonal part of the POVM and the whole POVM. This quantity might be interpreted as a
            measure of "non-classicality" or coherence present in measurement noise. See Ref. [1] for details.

        Parameters:
            -
        Return:
            -
        """

        self._coherent_error_bound = povmtools.operational_distance_POVMs(self._povm,
                                                                          povmtools.get_diagonal_povm_part(self._povm))

    @staticmethod
    def __calculate_statistical_error_bound(self, counts: np.ndarray) -> float:
        """
        Description:
            Get upper bound for tv-distance of estimated probability distribution from ideal one. See Ref. [3] for
            details.

        Parameters:
            :param counts: Counts for experiment for which statistical error bound is being calculated.

        Return:
            Statistical error upper bound in total variance distance.
        """

        number_of_measurement_outcomes = len(counts)
        number_of_samples = 0

        for count in counts:
            number_of_samples += count

        if number_of_measurement_outcomes < 16:
            # for small number of outcomes "-2" factor is not negligible
            return np.sqrt(
                (np.log(2 ** number_of_measurement_outcomes - 2)
                 - np.log(self.statistical_error_mistake_probability)) / 2 / number_of_samples
            )
            # for high number of outcomes "-2" factor is negligible
        else:
            return np.sqrt(
                (number_of_measurement_outcomes * np.log(2) - np.log(
                    self.statistical_error_mistake_probability)) / 2 / number_of_samples
            )

	

