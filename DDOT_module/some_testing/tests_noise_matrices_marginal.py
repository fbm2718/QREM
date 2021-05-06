"""
Created on 28.04.2021

@author: Filip Maciejewski
@contact: filip.b.maciejewski@gmail.com
"""

from povms_qi import povmtools
import numpy as np
from DDOT_module.child_classes.ddot_marginal_analyzer_vanilla import DDTMarginalsAnalyzer
import povms_qi.ancillary_functions as anf

A0 = povmtools.random_stochastic_matrix(2)
A12 = povmtools.random_stochastic_matrix(4)


A12_em = np.kron(np.eye(2),A12)
A1_em = np.kron(np.eye(2),np.kron(A0,np.eye(2)))

A02_em = povmtools.permute_matrix(A12_em,3,[1,2])

total_matrix = A02_em@A1_em

# total_matrix = np.kron(A0,A12)


# swap01 = np.kron(anf.swap(),np.eye(2))

# anf.ptr(swap01)
# total_matrix = povmtools.permute_matrix(total_matrix,3,[1,2])

big_N = 3
results_dictionary = {}

classical_register_big = ["{0:b}".format(i).zfill(big_N) for i in range(2 ** big_N)]

for i in range(int(2**big_N)):
    key_now = classical_register_big[i]
    distro_now = total_matrix[:, i]

    results_dictionary[key_now] = {"{0:b}".format(s).zfill(big_N): distro_now[s] for s in
                                   range(len(distro_now))}

# print(results_dictionary)



DDOT_analyzer_test = DDTMarginalsAnalyzer(results_dictionary,
                                          False)

subsets = [[i,j] for i in range(big_N) for j in range(i+1,big_N)]

DDOT_analyzer_test.compute_all_marginals(subsets)
#
# print(DDOT_analyzer_test._marginals_dictionary)
#
# raise KeyError
# lam0 = DDOT_analyzer_test.compute_noise_matrix_averaged([1])
# lam12 = DDOT_analyzer_test.compute_noise_matrix_averaged([0,2])

# lam0 = DDOT_analyzer_test.compute_noise_matrix_averaged([2])
# lam12 = DDOT_analyzer_test.compute_noise_matrix_averaged([0,1])

# anf.ptr(A0-lam0,10)
# anf.ptr(lam0)


# anf.ptr(A12-lam12,10)
# anf.ptr()


lam12_dep = DDOT_analyzer_test.compute_noise_matrix_dependent([0,1],[2])



# lam012 = DDOT_analyzer_test.get_noise_matrix_averaged([0,1,2])
# anf.ptr(lam012)
# anf.ptr(lam012-total_matrix)
#
# anf.ptr(lam12_dep['0']-lam12_dep['1'],10)
# anf.ptr(lam12_dep['00']-lam12_dep['01'],10)
# anf.ptr(lam12_dep['00']-lam12_dep['01'],10)
# anf.ptr(lam12_dep['00']-lam12_dep['10'],10)
# anf.ptr(lam12_dep['00']-lam12_dep['11'],10)
# print()


