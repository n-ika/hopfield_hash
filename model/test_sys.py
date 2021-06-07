from __future__ import print_function

import numpy as np
import random
import pandas as pd
from hash import *
from init_net import *
from _retrieve_memory import *

def precision_recall(T, V_train, V_test, N, U):
    """
    Perform a precision recall test on chosen files and parameters.

    Parameters
    ----------
    T : a numpy array T of shape (N, N)
        Memory in form of a matrix (synaptic weights)
    V_train : list of arrays (hashed features)
        The data we train our memory system on
    V_test : list of arrays (hashed features)
        The data we tast the memory system on
    N : num
        N=k*m, where k is the number of zeros in the hashed vector, m is number of ones
    U : num
        Chosen threshold above which the neuron is active
    Returns
    -------
    num
        We return values of hits, false alarms, correct rejections, misses
    """
    
    K = []
    correct_rejection = 0
    false_alarm = 0
    default = np.zeros(N)
    for v in V_test:
        memory, k = retrieve_memory(T, v, U=np.float(U))
        K.append(k)
        if np.array_equal(memory, default):
            correct_rejection += 1
        else:
            false_alarm += 1
    correct_rejection = correct_rejection/len(V_test)
    false_alarm = false_alarm/len(V_test)

    hits = 0
    miss = 0
    default = np.zeros(N)
    for v in V_train:
        memory, k = retrieve_memory(T, v, U=np.float(U))
        K.append(k)
        if np.array_equal(memory, default):
            miss += 1
        else:
            hits += 1
    hits = hits/len(V_train)
    miss = miss/len(V_train)
    
    return hits, false_alarm, correct_rejection, miss, K



def count_errors(initial_state, retrieved_state):
    equality_values = initial_state == retrieved_state
    count = np.count_nonzero(equality_values)
    return(len(equality_values)-count)


def calculate_distance(initial_state, retrieved_state, N, decision):
    equality_values = initial_state == retrieved_state
    count = np.count_nonzero(equality_values)
    dist = count / N
    if dist > decision:
        familiarity = 1
    else:
        familiarity = 0
    return(familiarity)

def test_memory(METHOD, TEST, mfccs_vectors=None, U = 0, N = 100, g = 100, p_list = [0.01, 0.1, 0.5], SEED = 27):

    n_list = [1, 2, int(N*0.05), int(N*0.1), int(N*0.15), int(N*0.5), N, N*2]
    
    if TEST == "errors":
        results = {"threshold":[], "N":[], "n":[], "errors":[], "p":[], "k":[], "type":[]}
    if TEST == "PR":
        results = {"threshold":[], "N":[], "n":[], "TPR":[], "FPR":[], "p":[], "k":[]}
    if TEST == "distance":
        results = {"threshold":[], "N":[], "n":[], "familiarity":[], "p":[], "k":[], "type":[]}
        
    for p in p_list:
        if mfccs_vectors == None:
            V = []
            for n in range(0,10000):
                rndm_vect = np.random.binomial(1, p, size=N)
                V.append(rndm_vect)

        else:
            k = int(1/p)
            assert k == 1/p
            m = int(N/k)
            assert m == N/k, (m,N/k)
            print("k, m: ", k, m)       
            V = hash_dataset(mfccs_vectors, k, m, SEED)

        for n in n_list:
            
            V_train = V[:n]
            V_test = V[n+1:n+n]
            
            if METHOD == "default":
                T_I = initialize_network(N, 0.5, V_train) # no inhibition
                U_eff = np.float(U)
            elif METHOD == "sparsity":
                T_I = initialize_network(N, p, V_train) # no inhibition
                U_eff = np.float(U)
            elif METHOD == "amits":
                T = initialize_network(N, p, V_train)            
    #             a=2*p-1
                T_I = T-g/N # with inhibition
                np.fill_diagonal(T_I, 0)
                U_eff = np.float(U)+g*(.5/N-p)
            elif METHOD == "tsodyks":
                T_I = initialize_network(N, p, V_train)  # no inhibition          
                U_eff = np.float(U)+N*p
            elif METHOD == "both":
                T = initialize_network(N, p, V_train)
                T_I = T-g/N # with inhibition
                U_eff = np.float(U)+N*p+g*(.5/N-p)

            # Test the system with the same vectors that are stored in memory
            # Check how many values of each state has changed
            if TEST == "PR":
                hit, fa, corr_rej, miss, K = precision_recall(T_I, V_train, V_test, N, U_eff)
                if hit + miss == 0:
                    TPR = np.NaN
                else:
                    TPR = hit / (hit + miss)
                if fa + corr_rej == 0:
                    FPR = np.NaN
                else:
                    FPR = fa / (fa + corr_rej)
                results["k"].append(K)
                results["TPR"].append(TPR)
                results["FPR"].append(FPR)

            if TEST == "errors" or TEST == "distance":

                for (i, memory_state) in enumerate(V_train):
                    retrieved_state, k = retrieve_memory(T_I, memory_state, U=U_eff)
                    results["k"].append(k)
                    if TEST == "errors":
                        counts = count_errors(memory_state, retrieved_state)
                        results["errors"].append(counts)
                    elif TEST == "distance":
                        dec_val = 100-p
                        dist = calculate_distance(memory_state, retrieved_state, N, dec_val)
                        results["familiarity"].append(dist)
                    results["type"].append("known")
                    results["threshold"].append(U_eff)
                    results["N"].append(N)
                    results["n"].append(n)
                    results["p"].append(p)

                for (i, memory_state) in enumerate(V_test):
                    retrieved_state, k = retrieve_memory(T_I, memory_state, U=U_eff)
                    results["k"].append(k)
                    if TEST == "errors":
                        counts = count_errors(memory_state, retrieved_state)
                        results["errors"].append(counts)
                    elif TEST == "distance":
                        dec_val = 100-p
                        dist = calculate_distance(memory_state, retrieved_state, N, dec_val)
                        results["familiarity"].append(dist)
                    results["type"].append("new")
                    results["threshold"].append(U_eff)
                    results["N"].append(N)
                    results["n"].append(n)
                    results["p"].append(p)

    results = pd.DataFrame.from_dict(results)   
    return(results)
