"""
Created on 01.03.2021

@author: Filip Maciejewski, Oskar Słowik
@contact: filip.b.maciejewski@gmail.com, osslowik@gmail.com
"""

class InfinityException(Exception):
    """Class for handling infinity"""
    pass
    
class ClusterSizeError(Error):
    """Class for handling infinity"""
    pass
    

  
#cluster size function. Assumes C_max>=3. If not, raises exception.
def f_clust(C_size, C_maxsize):
    if(C_max<3):
        raise ClusterSizeError
    val=0
    if C_size>C_maxsize:
        raise InfinityException
    elif C_size<3:
        val=0
    else:
        val=math.pow(C_size/C_maxsize,2)
    return val
   
#average inter-cluster attractivness function S_{i,j}
def S_ij_av(C_i,C_j,correlations_table):
    val=0
    for k in C_i:
        for l in C_j:
            val=val+correlations_table[k,l]/(len(C_i)*len(C_j))
return val

#intra-cluster attractivness function S_{i}
def S_i(C_i,correlations_table):
    val=0
    for k in C_i:
        for l in C_j:
            if(k==l):
                continue
            val=val+correlations_table[k,l]/(len(C_i)*len(C_i))
return val
    
    


#returns the value of cost function (simpler version; no badness). Raises error if minus infinity.
def cost_function_simple(partition, correlations_table, alpha, C_maxsize):
    val=0
    for C_i in partition:
        try:
            val=val-f_clust(len(C_i),C_maxsize)
        except:
            pass
     
    for C_i in partition:
        val=val+*alpha*S_i(C_i,correlations_table)
        
    for i in range(len(partition)):
        for j in range(len(partition)):
            if(j==i):
                continue
            C_i=partition[i]
            C_j=partition[j]
            val=val-alpha*S_ij_av(C_i,C_j,correlations_table)
    return val
            
#accepts (True) or rejects (False) move operation (qubit k from C_i to C_j) on clusters based on the change of cost_function_simple.
def evaluate_move_operation(partition, index_k, index_C_i, index_C_j, correlations_table, alpha, C_maxsize):
    diff=0
    C_i=partition[index_C_i]
    C_j=partition[index_C_j]
    k=C_i[index_k]
    try:
        diff=diff+f_clust(len(C_i),C_maxsize)+f_clust(len(C_j),C_maxsize)-f_clust(len(C_i)-1,C_maxsize)-f_clust(len(C_j)+1,C_maxsize)
    except ClusterSizeError:
        pass
    except InfinityException:
        return False
    for q in C_i:
        val=val-alpha*(correlations_table[k,q]+correlations_table[q,k])/(len(C_i)*len(C_i))
    for q in C_j:
        val=val+alpha*(correlations_table[k,q]+correlations_table[q,k])/(len(C_j)*len(C_j))
    #loops splitted to improve clarity. Can join them.
    for q in C_i:
        val=val-alpha*(correlations_table[k,q]+correlations_table[q,k])/(len(C_i)*len(C_i))
    for q in C_j:
        val=val+alpha*(correlations_table[k,q]+correlations_table[q,k])/(len(C_j)*len(C_j))
    
    
    
    
    
    