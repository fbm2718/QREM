"""
Created on 03.05.2021

@author: Filip Maciejewski
@contact: filip.b.maciejewski@gmail.com
"""

import numpy as np

from collections import Counter

circuits_labels_list = [[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 0]]

circuits_labels_strings_list = [''.join([str(symbol) for symbol in symbols_list]) for symbols_list in
                                circuits_labels_list]
same_circuits_counter = dict(zip(Counter(circuits_labels_strings_list).keys(),
                                 Counter(circuits_labels_strings_list).values()))



print(same_circuits_counter)
