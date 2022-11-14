from __future__ import print_function

import torch
import numpy as np
import random
import pandas as pd
from hash import *
from init_net import *
from retrieve_memory import *

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
    default = torch.zeros(N)
    for v in V_test:
        memory, k = retrieve_memory(T, v, U=np.float(U))
        K.append(k)
        if torch.equal(memory, default):
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
        if torch.equal(memory, default):
            miss += 1
        else:
            hits += 1
    hits = hits/len(V_train)
    miss = miss/len(V_train)
    
    return hits, false_alarm, correct_rejection, miss, K



def count_errors(initial_state, retrieved_state):
    equality_values = initial_state == retrieved_state
    count = torch.count_nonzero(equality_values)
    return(equality_values.shape[0]-count)


def calculate_distance(initial_state, retrieved_state, N, decision):
    equality_values = initial_state == retrieved_state
    count = torch.count_nonzero(equality_values)
    dist = count / N
    if dist > decision:
        familiarity = 1
    else:
        familiarity = 0
    return(familiarity)

def test_memory(METHOD, TEST, mfccs_vectors=None, U = 0, N = 100, n = 15, g = 100, p = 0.1, d=0.01, SEED = 27):
    
    if TEST == "errors":
        results = {"threshold":[], "N":[], "n":[], "errors":[], "p":[], "k":[], "type":[]}
    if TEST == "PR":
        results = {"threshold":[], "N":[], "n":[], "TPR":[], "FPR":[], "p":[], "k":[]}
    if TEST == "distance":
        results = {"threshold":[], "N":[], "n":[], "familiarity":[], "p":[], "k":[], "type":[]}
        
    # MAKE TRAINING DATA
    if mfccs_vectors == None:
        print("constructing stim...")
        rng = np.random.default_rng(SEED)
        active = int(N*p)
        non_active = int(N - N*p)
        V = np.array(rng.permutation([1]*active + [0]*non_active))
        for j in range(1,10000):
            rndm_vect = rng.permutation([1]*active + [0]*non_active)
            V = np.vstack((V, rndm_vect))
        V = torch.from_numpy(V)

    else:
        print("hashing mfccs...")
        k = int(1/p)
        assert k == 1/p
        m = int(N/k)
        assert m == N/k, (m,N/k)
        print("k, m: ", k, m)       
        V = hash_dataset(mfccs_vectors, k, m, SEED)
    
    # train with n memories and test on n noisy items    
    V_train = V[:n]

    # MAKE TEST DATA
    V_test = V_train.detach().clone()

    torch.manual_seed(SEED)
    for v in V_test:
        idxs = torch.randperm(N)[:int(N*d)]
        v[idxs] = torch.absolute(v[idxs]-1)
 
  
    # TRAIN SYSTEM
    if METHOD == "default":
        T_I = initialize_network(N, 0.5, V_train) # no inhibition
        U_eff = np.float(U)
    elif METHOD == "sparsity":
        T_I = initialize_network(N, p, V_train) # no inhibition
        U_eff = np.float(U)
    elif METHOD == "amit":
        T = initialize_network(N, p, V_train)            
#             a=2*p-1
        T_I = T-g/N # with inhibition
        T_I.fill_diagonal_(0)
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
                results["errors"].append(counts.item())
            elif TEST == "distance":
                dec_val = 100-p
                dist = calculate_distance(memory_state, retrieved_state, N, dec_val)
                results["familiarity"].append(dist)
            results["type"].append("known")
            results["threshold"].append(U_eff)
            results["N"].append(N)
            results["n"].append(n)
            results["p"].append(p)

        for (j, noisy_state) in enumerate(V_test):
            # test with noisy item
            # evaluate with memorized item
            memory_state = V_train[j]
            retrieved_state, k = retrieve_memory(T_I, noisy_state, U=U_eff)     
            results["k"].append(k)
            if TEST == "errors":
                counts = count_errors(memory_state, retrieved_state)
                results["errors"].append(counts.item())
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

# results_artif = test_memory('amit', 'errors', mfccs_vectors=None, U = 0, N = 100, n = 16, g = 1, p = 0.1, d = 0.05, SEED = 2)

