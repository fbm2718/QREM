"""
Created on 04.05.2021

@author: Filip Maciejewski
@contact: filip.b.maciejewski@gmail.com
"""


import pickle, os
import numpy as np
from QREM import ancillary_functions as anf
from QREM.DDOT_module.functions import functions_data_analysis as fda

module_directory = anf.get_module_directory()
tests_directory = module_directory + '/data_for_tests/'


example = fda.KeyDependentDictForMarginals()

dictionary_save = {'rick':example}

# data used for testing
backend_name = 'pickles'
date = '2020_05_04'

date_save = '2020_05_04'
directory = tests_directory + 'DDOT/' + backend_name + '/' + date_save + '/'
#
#
from povms_qi import povm_data_tools as pdt
pdt.Save_Results_simple(dictionary_save,
                        directory,
                        'test_results')


files = os.listdir(directory)
with open(directory + files[-1], 'rb') as filein:
    dictionary_data = pickle.load(filein)

example_unpickled = dictionary_data['rick']

# example_unpickled.

print(example_unpickled)
example_unpickled['q0q1q4q15']+=np.full((16,1),1/5)

print(example_unpickled)
# print(dictionary_data['god_damn_it'])








