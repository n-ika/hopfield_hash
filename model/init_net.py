import numpy as np

def initialize_network(N, p, V=None):
    """
    Eq. (2) from [1]
    
    Initialize synaptic weights in form of matrix T (symmetric recurrent weight matrix).
    This is a memory storage.
    
    Parameters
    ----------
    N : num
        number of neurons
    V : list
        list of vectors in a hash form
    p : num
        sparsity - probability of a value being 1
    Returns
    -------
    a numpy array T of shape (N, N)
        Memory storage in form of a matrix (synaptic weights)
        Its dimensions are determined by N=k*m (hash parameters)
    """
    
    T = np.zeros((N,N))
    if not(V is None):
        for vect in V:
            T = add_memory(T, vect, p)  
    return T


def add_memory(T, new_memory, p):
    """
    Eq. (2) from [1]
    
    Update synaptic weights in form of matrix T (symmetric recurrent weight matrix) when adding new memory.
    
    Parameters
    ----------
    T : a numpy array T_sum of shape (N, N)
        Initialized memory storage in form of a matrix (synaptic weights)
    new_memory : numpy array of shape (1,N)
        a vector we wish to store
    p : num
        a number representing the probability of a value being 1 (i.e. sparsity)
        
    Returns
    -------
    a numpy array T of shape (N, N)
        Renewed memory storage in form of a matrix (synaptic weights)
        Its dimensions are determined by N=k*m (hash parameters)
    """
    N = np.shape(T)[0]
#     if method == "default":
#         v = 2*new_memory - 1 #hopfield

    v = new_memory - p #tsodyks
#     elif method == "amits":
#         v = 2*new_memory - 2*p #amits
    outer_prod = np.outer(v,v)
    np.fill_diagonal(outer_prod, 0)
    T += outer_prod
        
    return T