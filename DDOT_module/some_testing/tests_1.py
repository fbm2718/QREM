"""
Created on 28.04.2021

@author: Filip Maciejewski
@contact: filip.b.maciejewski@gmail.com
"""



import numpy as np
qubits = [0,1,2]

print('q'.join([str(s) for s in qubits]))
print('q'+'q'.join([str(s) for s in qubits]))
# print('q'.join([s for s in qubits]))
from collections import defaultdict

from DDOT_module.functions import functions_data_analysis as fda

marginals_dict = fda.get_key_dependent_dict_for_marginals()

x = np.ones((4,1),dtype=float)

test_string = 'q2q3'
# test_length

marginals_dict['q2q3']+=x

marg_empty = defaultdict()


print(marginals_dict)
print(marginals_dict['q2q3'])
print(marg_empty)

for key in marginals_dict:
    print(key)


print()
z1 = np.array([12,25,37])
z2 = np.array([309,352,951])

# z2 = 2*z1

z12 = z1+z2

z12_sum = z1/sum(z1)+z2/sum(z2)

print(z12_sum/sum(z12_sum))
print(z12/sum(z12))


marg_empty = defaultdict(float)

marg_empty['00']+=6

print(marg_empty)




def fxsqr(a,b,c,x):
    print(a,b,c,x)
    return a*x**2+b*x+c


def pass_arguments_to_f(a,kwargs):

    return fxsqr(a=a,**kwargs)



kwargs_dict = {'c':0.2,'x':5,'b':0.1,}

print(pass_arguments_to_f(1,kwargs_dict))

print(sorted([1,2]))
print(sorted([1,2],reverse=True))





