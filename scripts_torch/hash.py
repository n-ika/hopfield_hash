# import numpy as np
import random
import torch

def hash_dim(d,k,m,seed):  

    """
    Define hash parameters.
    The hash will be a matrix of the dimension = k*m
    We choose a random number k of units of the vector.

    Parameters
    ----------
    d : num
        Length of a random vector being stored
    k : num
        Number of units we randomly choose of the vector
    m : num
        Number of times we will  do the hashing for some vector
    seed : num
        We always want the same units randomly chosen
        
    Returns
    -------
    a numpy array 
        p of dimensions [k,m] represents randomly chosen dimensions

    """   
    assert k <= d
    p = torch.zeros(m,k,)
    torch.manual_seed(seed)
    for i in range(m):
        p[i] = torch.randperm(d)[:k]
    return p

    
def get_hash(vector, k, m, p): 
    """
    Transform a vector of speech into a hash
    The hash will be a matrix of the dimension = k*m
    
    Once we have chosen k random dimensions, we look for the highest 
    value and turn it into 1. Everything else is 0.
    We thus get sparse matrices.
    We do this m times. Final output is h=k*m.
    
    Parameters
    ----------
    vector : torch tensor
        Features (i.e. MFCC) of some sound with dim = 1*n
    k : num
        Number of units we randomly choose of the vector
    m : num
        Number of times we will do the hashing for some vector.
    p : torch tensor
        p of dimensions [k,m] represents randomly chosen dimensions
        
    Returns
    -------
    a torch tensor h of size [1, k*m]
    """
    h = torch.zeros(m,k,)
    for i in range(m):
        p_line = torch.Tensor.int(p[i]).long() # torch tensors as idx need to be long
        ix = torch.argmax(vector[p_line])
        hi = torch.zeros(k)
        hi[ix] = 1
        h[i] = hi
    h = torch.hstack(h)
    return h


def hash_dataset(mfccs_vectors, k, m, SEED):
    """
    Make a hashed dataset with parameters k and m and with the extracted mfccs.

    Parameters
    ----------
    mfccs_vectors : torch tensor
        tensor of mfcc vector arrays extracted from an audio file, each array is a file
    k : num
        Number of units we randomly choose of the vector
    m : num
        Number of times we will do the hashing for some vector.
    Returns
    -------
    list
        We return a list of torch tensors, each representing a hashed audio file
    """
    d = len(mfccs_vectors[0])
    V =[]
    p = hash_dim(d,k,m,SEED).astype(int)
    for vect in mfccs_vectors:
        v = get_hash(vect, k, m, p)
        V.append(v) 
    return V






