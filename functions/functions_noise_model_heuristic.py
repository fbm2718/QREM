"""
@authors: Filip Maciejewski, Oskar Słowik, Tomek Rybotycki
@contact: filip.b.maciejewski@gmail.com
"""

import copy
import random
import math
import numpy as np
from functions import functions_data_visualization
from tqdm import tqdm

class InfinityException(Exception):
    """Class for handling infinity"""
    pass


class ClusterSizeError(NameError):
    """Class for handling too small max cluster size"""
    pass


# cluster size function. Assumes C_max>=3. If not, raises exception.
def f_clust_sharp(C_size, C_maxsize):
    # if(C_max<3):
    #    raise ClusterSizeError
    val = 0
    if C_size > C_maxsize:
        raise InfinityException
    return val


# cluster size function. Assumes C_max>=3. If not, raises exception.
def f_clust(C_size, C_maxsize):
    # if(C_max<3):
    #    raise ClusterSizeError
    val = 0
    if C_size > C_maxsize:
        raise InfinityException
    elif C_size < 3:
        val = 0
    else:
        val = math.pow(C_size / C_maxsize, 2)
    return val


# average inter-cluster attractivness function S_{i_index,j}
def get_S_ij_av(C_i, C_j, correlations_table):
    val = 0
    for k in C_i:
        for l in C_j:
            val = val + (correlations_table[k, l] + correlations_table[l, k]) / (
                    2 * len(C_i) * len(C_j))
    return val


# intra-cluster attractivness function S_{i_index}
def get_S_i(C_i, correlations_table):
    if len(C_i) < 2:
        return 0
    val = 0
    for k in C_i:
        for l in C_i:
            if (k == l):
                continue
            val = val + correlations_table[k, l] / (len(C_i) * (len(C_i) - 1))
    return val


# intra-cluster cumulative attractivness function Sc_{i_index}
def get_Sc_i(C_i, correlations_table):
    if len(C_i) < 2:
        return 0
    val = 0
    for k in C_i:
        for l in C_i:
            if (k == l):
                continue
            val = val + correlations_table[k, l]
    return val


def cost_function_simple_cummulative(partition, correlations_table, alpha, C_maxsize):
    no_of_clusters = len(partition)
    val = 0
    for C_i in partition:
        try:
            val = val - f_clust_sharp(len(C_i), C_maxsize)
        except InfinityException:
            raise InfinityException

    for i in range(no_of_clusters):
        C_i = partition[i]
        S_i = get_Sc_i(C_i, correlations_table)
        val = val + alpha * S_i
    return val


# returns the value of cost function (simpler method_name; no badness) and S table - potentially_stochastic_matrix symmetric matrix which is defined by S_ij=S_ij_av (off-diagonal) and S_ii=S_i (diagonal). Raises error if minus infinity.
def cost_function_simple(partition, correlations_table, alpha, C_maxsize):
    no_of_clusters = len(partition)
    S = np.zeros((no_of_clusters, no_of_clusters))
    val = 0
    for C_i in partition:
        try:
            val = val - f_clust(len(C_i), C_maxsize)
        except:
            raise InfinityException

    for i in range(no_of_clusters):
        C_i = partition[i]
        S_i = get_S_i(C_i, correlations_table)
        S[i, i] = S_i
        val = val + alpha * S_i

    for i in range(no_of_clusters - 1):
        for j in range(i + 1, no_of_clusters):
            C_i = partition[i]
            C_j = partition[j]
            S_ij = get_S_ij_av(C_i, C_j, correlations_table)
            S[i, j] = S_ij
            S[j, i] = S_ij
            val = val - alpha * S_ij
    return val, S


def evaluate_move_operation_naive_cummulative(partition, index_k, index_C_i, index_C_j,
                                              correlations_table, alpha, C_maxsize):
    partition_copy = copy.deepcopy(partition)
    val1 = cost_function_simple_cummulative(partition_copy, correlations_table, alpha, C_maxsize)
    make_move_operation(partition_copy, index_k, index_C_i, index_C_j)
    # print(S_1)
    try:
        val2 = cost_function_simple_cummulative(partition_copy, correlations_table, alpha, C_maxsize)
    except InfinityException:
        return 0, False
    diff = val2 - val1
    return diff, True


def evaluate_move_operation_naive(partition, index_k, index_C_i, index_C_j, correlations_table, alpha,
                                  C_maxsize, S):
    partition_copy = copy.deepcopy(partition)
    val1, S_1 = cost_function_simple(partition_copy, correlations_table, alpha, C_maxsize)
    # print(S_1)
    if len(partition_copy[index_C_i]) < 2:
        print("deletion")
        S_1 = np.delete(S_1, index_C_i, 0)
        S_1 = np.delete(S_1, index_C_i, 1)

    make_move_operation(partition_copy, index_k, index_C_i, index_C_j)
    # print(S_1)
    try:
        val2, S_2 = cost_function_simple(partition_copy, correlations_table, alpha, C_maxsize)
    except InfinityException:
        return 0, False
    diff = val2 - val1
    dS = S_2 - S_1
    print(S_2)
    return diff, dS, True


def evaluate_swap_operation_naive_cummulative(partition, index_k, index_l, index_C_i, index_C_j,
                                              correlations_table, alpha, C_maxsize):
    partition_copy = copy.deepcopy(partition)
    val1 = cost_function_simple_cummulative(partition_copy, correlations_table, alpha, C_maxsize)
    make_swap_operation(partition_copy, index_k, index_l, index_C_i, index_C_j)
    val2 = cost_function_simple_cummulative(partition_copy, correlations_table, alpha, C_maxsize)
    diff = val2 - val1
    return diff


def evaluate_swap_operation_naive(partition, index_k, index_l, index_C_i, index_C_j,
                                  correlations_table, alpha, C_maxsize, S):
    partition_copy = copy.deepcopy(partition)
    val1, S_1 = cost_function_simple(partition_copy, correlations_table, alpha, C_maxsize)
    make_swap_operation(partition_copy, index_k, index_l, index_C_i, index_C_j)
    val2, S_2 = cost_function_simple(partition_copy, correlations_table, alpha, C_maxsize)
    diff = val2 - val1
    dS = S_2 - S_1
    return diff, dS


# returns the change of cost function cost_function_simple after potentially_stochastic_matrix move operation (qubit k from C_i to C_j) on clusters.
def evaluate_move_operation(partition, index_k, index_C_i, index_C_j, correlations_table, alpha,
                            C_maxsize, S):
    no_of_clusters = len(partition)
    dS = np.zeros((no_of_clusters, no_of_clusters))
    diff = 0
    C_i = partition[index_C_i]
    C_j = partition[index_C_j]
    c_i = len(C_i)
    c_j = len(C_j)

    # intra attraction part
    dS_intra = 0

    if c_i > 2:
        psum = 0
        for index_q in C_i:
            if index_q == index_k:
                continue
            psum = psum + correlations_table[index_k][index_q] + correlations_table[index_q][index_k]
        psum = -psum / ((c_i - 1) * (c_i - 2))
        psum = psum + (c_i * (c_i - 1) / ((c_i - 1) * (c_i - 2)) - 1) * S[index_C_i, index_C_i]
    else:
        psum = -S[index_C_i, index_C_i]

    dS[index_C_i, index_C_i] = psum
    dS_intra = dS_intra + psum

    psum = 0
    sum_k_q_C_j = 0
    for index_q in C_j:
        psum = psum + correlations_table[index_k][index_q] + correlations_table[index_q][index_k]
    sum_k_q_C_j = psum
    psum = psum / ((c_j + 1) * c_j)
    psum = psum + (c_j * (c_j - 1) / ((c_j + 1) * c_j) - 1) * S[index_C_j, index_C_j]
    dS[index_C_j, index_C_j] = psum
    dS_intra = dS_intra + psum

    # inter attraction part
    dS_inter = 0
    for index_C_m in range(len(partition)):
        if index_C_m in [index_C_i, index_C_j]:
            continue
        C_m = partition[index_C_m]
        c_m = len(C_m)

        psum = 0
        for index_q in C_m:
            psum = psum + correlations_table[index_k][index_q] + correlations_table[index_q][index_k]
        if c_i > 1:
            psum1 = (-1 / (2 * c_m)) * (1 / (c_i - 1)) * psum + (c_i * c_m / ((c_i - 1) * c_m) - 1) * \
                    S[index_C_i, index_C_m]
        else:
            psum1 = -S[index_C_i, index_C_m]
        dS[index_C_i, index_C_m] = psum1
        dS[index_C_m, index_C_i] = psum1
        psum2 = (1 / (2 * c_m)) * (1 / (c_j + 1)) * psum + (c_j * c_m / ((c_j + 1) * c_m) - 1) * S[
            index_C_j, index_C_m]
        dS[index_C_j, index_C_m] = psum2
        dS[index_C_m, index_C_j] = psum2
        dS_inter = dS_inter + psum1 + psum2

    psum = 0
    if c_i > 1:
        for index_q in C_i:
            if index_q == index_k:
                continue
            psum = psum + correlations_table[index_k][index_q] + correlations_table[index_q][index_k]
        psum = (1 / (2 * (c_i - 1) * (c_j + 1))) * (psum - sum_k_q_C_j) + (
                c_i * c_j / ((c_i - 1) * (c_j + 1)) - 1) * S[index_C_i, index_C_j]
    else:
        psum = -S[index_C_i, index_C_j]

    dS[index_C_i, index_C_j] = psum
    dS[index_C_j, index_C_i] = psum
    dS_inter = dS_inter + psum

    # cluster function part
    dS_clust = 0
    try:
        dS_clust = f_clust(len(C_i) - 1, C_maxsize) + f_clust(len(C_j) + 1, C_maxsize) - f_clust(
            len(C_i), C_maxsize) - f_clust(len(C_j), C_maxsize)
    except InfinityException:
        return diff, dS, False

    diff = alpha * (dS_intra - dS_inter) - dS_clust

    return diff, dS, True


# returns the change of cost function cost_function_simple after potentially_stochastic_matrix swap operation (qubit k from C_i wilh qubit l_index from C_j) on clusters.
# TODO: values do not match the naive method_name - find an error!
def evaluate_swap_operation(partition, index_k, index_l, index_C_i, index_C_j, correlations_table,
                            alpha, C_maxsize, S):
    # cluster function part - contrubutes zero
    # intra attraction part
    no_of_clusters = len(partition)
    dS = np.zeros((no_of_clusters, no_of_clusters))
    dS_intra = 0
    C_i = partition[index_C_i]
    C_j = partition[index_C_j]
    c_i = len(C_i)
    c_j = len(C_j)

    if c_i > 1:
        psum = 0
        for index_q in C_i:
            if index_q == index_k:
                continue
            psum = psum + correlations_table[index_q][index_l] - correlations_table[index_q][index_k]
        psum = psum * (1 / (c_i * (c_i - 1)))
        dS[index_C_i, index_C_i] = psum
        dS_intra = dS_intra + psum

    if c_j > 1:
        psum = 0
        for index_q in C_j:
            if index_q == index_l:
                continue
            psum = psum + correlations_table[index_q][index_k] - correlations_table[index_q][index_l]
        dS[index_C_j][index_C_j] = psum
        psum = psum * (1 / (c_j * (c_j - 1)))
        dS_intra = dS_intra + psum

    # inter attraction part
    dS_inter = 0
    for index_C_m in range(len(partition)):
        if index_C_m in [index_C_i, index_C_j]:
            continue
        C_m = partition[index_C_m]
        c_m = len(C_m)

        psum = 0
        for index_q in C_m:
            psum = psum + correlations_table[index_l][index_q] - correlations_table[index_k][index_q]
        psum1 = (1 / (2 * c_m)) * (1 / c_i) * psum
        psum2 = (1 / (2 * c_m)) * (-1 / c_j) * psum
        dS[index_C_i, index_C_m] = psum1
        dS[index_C_m, index_C_i] = psum1
        dS[index_C_j, index_C_m] = psum2
        dS[index_C_m, index_C_j] = psum2
        dS_inter = dS_inter + psum1 + psum2

    psum = 0
    for index_q in C_j:
        if index_q == index_l:
            continue
        psum = psum + correlations_table[index_l][index_q] - correlations_table[index_k][index_q]
    for index_q in C_i:
        if index_q == index_k:
            continue
        psum = psum - (correlations_table[index_l][index_q] - correlations_table[index_k][index_q])
    psum = psum * (1 / (2 * c_i * c_j))
    dS[index_C_i, index_C_j] = psum
    dS[index_C_j, index_C_i] = psum
    dS_inter = dS_inter + psum

    diff = alpha * (dS_intra - dS_inter)
    return alpha * diff, dS


def get_initial_partition(correlations_table):
    n = correlations_table.shape[0]
    CL = dict()
    for i in range(n - 1):
        for j in range(i + 1, n):
            val = 0
            val1 = correlations_table[i, j]
            val2 = correlations_table[j, i]
            if (val1 > val2):
                val = val1
            else:
                val = val2
            CL.update({(i, j): val})
    # print(CL)
    partition = []
    qubits = list(range(n))
    while len(qubits) > 1:
        pair = max(CL, key=lambda key: CL[key])
        i = pair[0]
        j = pair[1]
        # print(i_index,j)
        partition.append([i, j])
        keys = list(CL.keys())
        # print(keys)
        for pair in keys:
            # print(str(pair)+"keys:"+str(i_index)+str(j))
            if pair[0] in [i, j] or pair[1] in [i, j]:
                # print("popping"+str(pair))
                CL.pop(pair, None)
        qubits.remove(i)
        qubits.remove(j)
    if len(qubits) == 1:
        partition.append([qubits[0]])

    return partition


def return_cluster_index(partition, target_qubit):
    index = 0
    for cluster in partition:
        for qubit in cluster:
            if qubit == target_qubit:
                return index
        index = index + 1
    return index


def make_move_operation(partition, index_k, index_C_i, index_C_j):
    # partition[index_C_i].remove(index_k)
    partition[index_C_j].append(index_k)

    if len(partition[index_C_i]) == 1:
        partition.pop(index_C_i)
    else:
        partition[index_C_i].remove(index_k)

    return


def make_swap_operation(partition, index_k, index_l, index_C_i, index_C_j):
    partition[index_C_i].remove(index_k)
    partition[index_C_i].append(index_l)
    partition[index_C_j].remove(index_l)
    partition[index_C_j].append(index_k)

    return


def partition_algorithm_v1(correlations_table, alpha, C_maxsize, N_alg):
    if (C_maxsize < 3):
        print("Error: max cluster size has to be at least 3!. Algorithm terminated.")
        return
    initial_partition = get_initial_partition(correlations_table)
    print(initial_partition)
    functions_data_visualization.print_partition(initial_partition)
    init_cf, init_S = cost_function_simple(initial_partition, correlations_table, alpha, C_maxsize)
    print("initial value: " + str(init_cf))
    no_of_qubits = correlations_table.shape[0]
    # results=dict()
    global_best_parition = initial_partition
    global_best_value = init_cf
    for attempt_no in range(N_alg):
        print("attempt: " + str(attempt_no + 1))
        partition = copy.deepcopy(initial_partition)
        best_cf = init_cf - 1
        curr_cf = init_cf
        curr_S = init_S
        epoch_no = 0
        while curr_cf > best_cf:
            epoch_no = epoch_no + 1
            best_cf = curr_cf
            print("starting epoch: " + str(epoch_no))
            pairs = []
            for i in range(no_of_qubits - 1):
                for j in range(i + 1, no_of_qubits):
                    pairs.append([i, j])
            while len(pairs) > 0:
                val_of_ops = dict()
                pair = random.choice(pairs)
                pairs.remove(pair)
                i = pair[0]
                j = pair[1]
                print("pair: " + "(" + str(i) + ", " + str(j) + ")")
                index_C_i = return_cluster_index(partition, i)
                index_C_j = return_cluster_index(partition, j)
                if (index_C_i == index_C_j):
                    print("WRONG PAIR - SKIP")
                    continue
                index_k = i
                val1, dS1, not_neg_infty = evaluate_move_operation(partition, index_k, index_C_i,
                                                                   index_C_j, correlations_table,
                                                                   alpha, C_maxsize, curr_S)
                val1_true, dS1_true, not_neg_infty_true = evaluate_move_operation_naive(partition,
                                                                                        index_k,
                                                                                        index_C_i,
                                                                                        index_C_j,
                                                                                        correlations_table,
                                                                                        alpha,
                                                                                        C_maxsize,
                                                                                        curr_S)

                print("checking move " + str(index_C_i) + "--" + str(index_k) + "->" + str(
                    index_C_j) + "\t, result: " + "{:.5f}".format(val1) + "\t, size_leq_C_max: " + str(
                    not_neg_infty) + "\t, true_res:" + "{:.5f}".format(val1_true))
                if val1 > 0 and not_neg_infty:
                    print("update")
                    val_of_ops.update({'move_ij': val1})
                index_C_i = return_cluster_index(partition, j)
                index_C_j = return_cluster_index(partition, i)
                index_k = j
                val2, dS2, not_neg_infty = evaluate_move_operation(partition, index_k, index_C_i,
                                                                   index_C_j, correlations_table,
                                                                   alpha, C_maxsize, curr_S)
                val2_true, dS2_true, not_neg_infty_true = evaluate_move_operation_naive(partition,
                                                                                        index_k,
                                                                                        index_C_i,
                                                                                        index_C_j,
                                                                                        correlations_table,
                                                                                        alpha,
                                                                                        C_maxsize,
                                                                                        curr_S)
                print("checking move " + str(index_C_i) + "--" + str(index_k) + "->" + str(
                    index_C_j) + "\t, result: " + "{:.5f}".format(val2) + "\t, size_leq_C_max: " + str(
                    not_neg_infty) + "\t, true_res:" + "{:.5f}".format(val2_true))
                if val2 > 0 and not_neg_infty:
                    val_of_ops.update({'move_ji': val2})
                index_C_i = return_cluster_index(partition, i)
                index_C_j = return_cluster_index(partition, j)
                index_k = i
                index_l = j
                val3, dS3 = evaluate_swap_operation(partition, index_k, index_l, index_C_i, index_C_j,
                                                    correlations_table, alpha, C_maxsize, curr_S)
                val3_true, dS3_true = evaluate_swap_operation_naive(partition, index_k, index_l,
                                                                    index_C_i, index_C_j,
                                                                    correlations_table, alpha,
                                                                    C_maxsize, curr_S)
                print("checking swap " + str(index_C_i) + "<-" + str(index_k) + "&" + str(
                    index_l) + "->" + str(index_C_j) + "\t, result: " + "{:.5f}".format(
                    val3) + "\t, size_leq_C_max: " + str(
                    not_neg_infty) + "\t, true_res:" + "{:.5f}".format(val3_true))
                if val3 > 0:
                    val_of_ops.update({'swap': val3})

                if len(val_of_ops) > 0:
                    print("ACCEPT:")
                    op = max(val_of_ops, key=lambda key: val_of_ops[key])
                    if (op == 'move_ij'):
                        dS = dS1
                        index_C_i = return_cluster_index(partition, i)
                        index_C_j = return_cluster_index(partition, j)
                        index_k = i
                        c_i = len(partition[index_C_i])
                        print("MOVE_ij")
                        make_move_operation(partition, index_k, index_C_i, index_C_j)
                        print(partition)
                        curr_cf = curr_cf + val_of_ops[op]
                        curr_S = curr_S + dS
                        if c_i < 2:
                            curr_S = np.delete(curr_S, index_C_i, 0)
                            curr_S = np.delete(curr_S, index_C_i, 1)
                    elif (op == 'move_ji'):
                        dS = dS2
                        index_C_i = return_cluster_index(partition, j)
                        index_C_j = return_cluster_index(partition, i)
                        index_k = j
                        c_i = len(partition[index_C_i])
                        print("MOVE_ji")
                        make_move_operation(partition, index_k, index_C_i, index_C_j)
                        print(partition)
                        curr_cf = curr_cf + val_of_ops[op]
                        curr_S = curr_S + dS
                        if c_i < 2:
                            curr_S = np.delete(curr_S, index_C_i, 0)
                            curr_S = np.delete(curr_S, index_C_i, 1)
                    else:
                        dS = dS3
                        index_C_i = return_cluster_index(partition, i)
                        index_C_j = return_cluster_index(partition, j)
                        index_k = i
                        index_l = j
                        print("SWAP")
                        make_swap_operation(partition, index_k, index_l, index_C_i, index_C_j)
                        print(partition)
                        curr_cf = curr_cf + val_of_ops[op]
                        curr_S = curr_S + dS
                else:
                    print("REJECT ALL")
        print("convergence to: " + str(best_cf) + " after " + str(epoch_no) + " epochs")
        if (best_cf > global_best_value):
            global_best_parition = partition
            global_best_value = best_cf
        # results.update({partition:best_cf})
    # best_attempt=op=max(results, key=lambda key: results[key])
    return global_best_parition, global_best_value





def partition_algorithm_v1_cummulative(correlations_table,
                                       alpha,
                                       C_maxsize,
                                       N_alg,
                                       printing,
                                       drawing):
    if (C_maxsize < 2):
        print("Error: max cluster size has to be at least 2 ! Algorithm terminated.")
        return

    initial_partition = get_initial_partition(correlations_table)
    print("initial partition:")
    print(initial_partition)
    if drawing:
        functions_data_visualization.print_partition(initial_partition)
    init_cf = cost_function_simple_cummulative(initial_partition, correlations_table, alpha, C_maxsize)
    print("initial value: " + str(init_cf))
    if C_maxsize == 2:
        return initial_partition, init_cf

    no_of_qubits = correlations_table.shape[0]
    # results=dict()
    global_best_parition = initial_partition
    global_best_value = init_cf
    for attempt_no in tqdm(range(N_alg)):
        if (printing):
            print("attempt: " + str(attempt_no + 1))
        partition = copy.deepcopy(initial_partition)
        best_cf = init_cf - 1
        curr_cf = init_cf
        epoch_no = 0
        while curr_cf > best_cf:
            epoch_no = epoch_no + 1
            best_cf = curr_cf
            if (printing):
                print("starting epoch: " + str(epoch_no))
            pairs = []
            for i in range(no_of_qubits - 1):
                for j in range(i + 1, no_of_qubits):
                    pairs.append([i, j])
            while len(pairs) > 0:
                val_of_ops = dict()
                pair = random.choice(pairs)
                pairs.remove(pair)
                i = pair[0]
                j = pair[1]
                if (printing):
                    print("pair: " + "(" + str(i) + ", " + str(j) + ")")
                index_C_i = return_cluster_index(partition, i)
                index_C_j = return_cluster_index(partition, j)
                if (index_C_i == index_C_j):
                    if (printing):
                        print("WRONG PAIR - SKIP")
                    continue
                index_k = i
                # val1, dS1, not_neg_infty=evaluate_move_operation(partition, index_k, index_C_i, index_C_j, correlations_table_quantum, alpha, C_maxsize, curr_S)
                val1, not_neg_infty = evaluate_move_operation_naive_cummulative(partition, index_k,
                                                                                index_C_i, index_C_j,
                                                                                correlations_table,
                                                                                alpha, C_maxsize)
                if (printing):
                    print("checking move " + str(index_C_i) + "--" + str(index_k) + "->" + str(
                        index_C_j) + "\t, result: " + "{:.5f}".format(
                        val1) + "\t, size_leq_C_max: " + str(not_neg_infty))
                if val1 > 0 and not_neg_infty:
                    val_of_ops.update({'move_ij': val1})
                index_C_i = return_cluster_index(partition, j)
                index_C_j = return_cluster_index(partition, i)
                index_k = j
                # val2, dS2, not_neg_infty=evaluate_move_operation(partition, index_k, index_C_i, index_C_j, correlations_table_quantum, alpha, C_maxsize, curr_S)
                val2, not_neg_infty = evaluate_move_operation_naive_cummulative(partition, index_k,
                                                                                index_C_i, index_C_j,
                                                                                correlations_table,
                                                                                alpha, C_maxsize)
                if (printing):
                    print("checking move " + str(index_C_i) + "--" + str(index_k) + "->" + str(
                        index_C_j) + "\t, result: " + "{:.5f}".format(
                        val2) + "\t, size_leq_C_max: " + str(not_neg_infty))
                if val2 > 0 and not_neg_infty:
                    val_of_ops.update({'move_ji': val2})
                index_C_i = return_cluster_index(partition, i)
                index_C_j = return_cluster_index(partition, j)
                index_k = i
                index_l = j
                # val3, dS3=evaluate_swap_operation(partition, index_k, index_l, index_C_i, index_C_j, correlations_table_quantum, alpha, C_maxsize, curr_S)
                val3 = evaluate_swap_operation_naive_cummulative(partition, index_k, index_l,
                                                                 index_C_i, index_C_j,
                                                                 correlations_table, alpha, C_maxsize)
                if (printing):
                    print("checking swap " + str(index_C_i) + "<-" + str(index_k) + "&" + str(
                        index_l) + "->" + str(index_C_j) + "\t, result: " + "{:.5f}".format(
                        val3) + "\t, size_leq_C_max: " + str(not_neg_infty))
                if val3 > 0:
                    val_of_ops.update({'swap': val3})

                if len(val_of_ops) > 0:
                    if (printing):
                        print("ACCEPT:")
                    op = max(val_of_ops, key=lambda key: val_of_ops[key])
                    if (op == 'move_ij'):
                        index_C_i = return_cluster_index(partition, i)
                        index_C_j = return_cluster_index(partition, j)
                        index_k = i
                        make_move_operation(partition, index_k, index_C_i, index_C_j)
                        if (printing):
                            print("MOVE_ij")
                            print(partition)
                        curr_cf = curr_cf + val_of_ops[op]
                    elif (op == 'move_ji'):
                        index_C_i = return_cluster_index(partition, j)
                        index_C_j = return_cluster_index(partition, i)
                        index_k = j
                        make_move_operation(partition, index_k, index_C_i, index_C_j)
                        if (printing):
                            print("MOVE_ji")
                            print(partition)
                        curr_cf = curr_cf + val_of_ops[op]
                    else:
                        index_C_i = return_cluster_index(partition, i)
                        index_C_j = return_cluster_index(partition, j)
                        index_k = i
                        index_l = j
                        make_swap_operation(partition, index_k, index_l, index_C_i, index_C_j)
                        if (printing):
                            print("SWAP")
                            print(partition)
                        curr_cf = curr_cf + val_of_ops[op]
                elif (printing):
                    print("REJECT ALL")
        if (printing):
            print("convergence to: " + str(best_cf) + " after " + str(epoch_no) + " epochs")
        if (best_cf > global_best_value):
            global_best_parition = partition
            global_best_value = best_cf
        # results.update({partition:best_cf})
    # best_attempt=op=max(results, key=lambda key: results[key])

    global_best_parition_sorted = [sorted(cluster) for cluster in global_best_parition]
    return global_best_parition_sorted, global_best_value


# TESTING
"""
printing=False
drawing=False

C=10
alpha=1
C_maxsize=6
N_alg=1000
noise=0.3

correlations_table_quantum=np.full((10, 10), noise)

for i_index in range(10):
    correlations_table_quantum[i_index][i_index]=0

correlations_table_quantum[0][1]=0.9
correlations_table_quantum[1][0]=0.9
correlations_table_quantum[0][2]=0.9
correlations_table_quantum[2][0]=0.9
correlations_table_quantum[1][2]=0.9
correlations_table_quantum[2][1]=0.9
correlations_table_quantum[3][4]=0.7
correlations_table_quantum[4][3]=0.7
correlations_table_quantum[3][5]=0.7
correlations_table_quantum[5][3]=0.7
correlations_table_quantum[4][5]=0.7
correlations_table_quantum[5][4]=0.7
correlations_table_quantum[6][7]=0.5
correlations_table_quantum[7][6]=0.5
correlations_table_quantum[8][9]=0.4
correlations_table_quantum[9][8]=0.4

correlations_table_quantum=correlations_table_quantum/C

if(printing):
    print("correlations table:")
    print(correlations_table_quantum)

partition, score=partition_algorithm_v1_cummulative(correlations_table_quantum, alpha, C_maxsize, N_alg, printing, drawing)

print("final partition:")
print(partition)
print("final value: "+str(score))
if(drawing):
    functions_data_visualization.print_partition(partition)
"""
