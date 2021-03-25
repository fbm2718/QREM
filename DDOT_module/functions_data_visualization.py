"""
Module for cluster visualization. Requires networkx.
Created on 25.03.2021

@author: Oskar Słowik
@contact: osslowik@gmail.com
"""

import networkx as nx
from itertools import chain


A=[[1,2,3,4],[5,6,7],[8,9],[10]]


def create_undirected_nxgraph_from_partition(partition):
    q_list=[q for C in partition for q in C]
    G=nx.DiGraph()
    G.add_nodes_from(q_list)
    for C in partition:
        for i in range(len(C)-1):
            for j in chain(range(i),range(i+1,len(C))):
                G.add_edge(C[i],C[j])
    return G
 
#prints clusters with edges only within a cluster 
def print_partition(partition):
    G=create_undirected_nxgraph_from_partition(partition)
    nx.draw_planar(G)
    
print_partition(A)   
    