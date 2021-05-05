"""
Created on 05.05.2021

@author: Filip Maciejewski
@contact: filip.b.maciejewski@gmail.com
"""

from QREM.DDOT_module.examples.generating_artificial_data import helper_functions as hpf
from QREM import ancillary_functions as anf
import numpy as np
from QREM.DDOT_module.child_classes.global_noise_matrix_creator import GlobalNoiseMatrixCreator
from QREM.DDOT_module.child_classes.noise_model_generator_vanilla import NoiseModelGenerator

# specify range of noise parameters

# probability of getting 1 if input was |0> will be in this range
p10_range = [0.005, 0.02]
# probability of getting 0 if input was |1>
p01_range = [0.015, 0.1]

# possible range of deviations of error probabilities dependeding on the state of the neighbour
# deviation here is in absolute numbers
deviation_p10_range = [p10_range[0] * 0.25, p10_range[1] * 0.5]
deviation_p01_range = [p01_range[0] * 0.75, p01_range[1] * 1]

neighbors_number_multiplier = 1.25

number_of_qubits = 3
list_of_qubits = list(range(number_of_qubits))
number_of_neighbours_range = [0, int(np.min([number_of_qubits - 1, 3]))]
# number_of_neighbours_range = [0,0]

clusters_list_true = [[qi] for qi in list_of_qubits]
noise_matrices_dictionary, neighbors_of_clusters = {}, []

for qubit_index in range(number_of_qubits):
    number_of_neighbours_now = int(
        np.random.uniform(number_of_neighbours_range[0], number_of_neighbours_range[1] + 1))
    neighbours_now = []

    if number_of_neighbours_now != 0:
        while len(neighbours_now) != number_of_neighbours_now:
            random_qubit = np.random.choice(list(set(list_of_qubits).difference(set([qubit_index]))))
            if random_qubit not in neighbours_now:
                neighbours_now.append(random_qubit)

        neighbours_now = sorted(neighbours_now)

        possible_states_neighbors = anf.register_names_qubits(range(number_of_neighbours_now),
                                                              number_of_neighbours_now)

        multiplier_now = neighbors_number_multiplier ** (1 - number_of_neighbours_now)

        p10_range_magnified, p01_range_magnified = [multiplier_now * pi for pi in p10_range], [
            multiplier_now * pi for pi in p01_range]

        single_qubit_noise_matrix_base = hpf.get_random_stochastic_matrix_1q(p10_range_magnified,
                                                                             p01_range_magnified)

        perturbed_noise_matrices = [
            hpf.perturb_stochastic_matrix_1q(single_qubit_noise_matrix_base,
                                             deviation_p10_range,
                                             deviation_p01_range) for neighbors_state_index in
            range(len(possible_states_neighbors))]

        state_dependent_noise_matrices = [single_qubit_noise_matrix_base] + perturbed_noise_matrices

        noise_matrices_dictionary_now = {}

        for neighbors_state_index in range(len(possible_states_neighbors)):
            neighbors_state_now = possible_states_neighbors[neighbors_state_index]

            noise_matrices_dictionary_now[neighbors_state_now] = state_dependent_noise_matrices[
                neighbors_state_index]

        noise_matrices_dictionary['q%s' % qubit_index] = noise_matrices_dictionary_now
        neighbors_of_clusters.append(neighbours_now)

    else:
        single_qubit_noise_matrix = hpf.get_random_stochastic_matrix_1q(p10_range, p01_range)
        noise_matrices_dictionary['q%s' % qubit_index] = {'averaged': single_qubit_noise_matrix}
        neighbors_of_clusters.append(None)

global_noise_creator = GlobalNoiseMatrixCreator(noise_matrices_dictionary,
                                                )

global_noise_matrix = global_noise_creator.compute_global_noise_matrix(clusters_list_true,
                                                                       neighbors_of_clusters)
# 
# print(noise_matrices_dictionary)


anf.print_array_nicely(global_noise_matrix, 3)
anf.print_array_nicely(np.diag(global_noise_matrix))
print(neighbors_of_clusters)

number_of_shots = 8192

classical_states = anf.register_names_qubits(range(number_of_qubits), number_of_qubits)

results_dictionary = {}

for bitstring_input in classical_states:

    counts_dictionary = {}

    probability_distribution_now = global_noise_matrix[:, int(bitstring_input, 2)]
    for bitstring_output in classical_states:
        counts_dictionary[bitstring_output] = probability_distribution_now[
                                                  int(bitstring_output, 2)] * number_of_shots




noise_model_analyzer = NoiseModelGenerator(results_dictionary_ddot=results_dictionary,
                                           reverse_counts=False,
                                           number_of_qubits=number_of_qubits)

# noise_model_analyzer
