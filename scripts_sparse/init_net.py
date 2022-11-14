# import torch
import numpy as np

def sparsify_matrix(N,s=0.4):
    """
    Make a sparse matrix to be used as a mask to sparsify 
    synaptic weight matrix - set some synaptic weights to 0 and preserve others
    Parameters
    ----------
    N : int
        number of neurons
    s : float
        sparsity degree of the matrix, num of active values is N*s
    Returns
    -------
    mask: np matrix
        a sparse matrix defining which values of the synaptic weight 
        matrix will be active (1) or inactive (0)

    """
    mask = np.vstack(np.array_split([(1 if float(np.random.rand(1)) < s else 0) for x in range(N*N)],N))
    upper = np.triu(mask)
    mask = upper+upper.transpose()
    np.fill_diagonal(mask, 0)
    return(mask)


def initialize_network(N, p, mask, V=None):
    """
    Eq. (2) from [1]
    
    Initialize synaptic weights in form of matrix T (symmetric recurrent weight matrix).
    This is a memory storage.
    
    Parameters
    ----------
    N : int
        number of neurons
    V : list
        list of vectors in a hash form
    p : float
        sparsity - probability of a value being 1
    Returns
    -------
    a np matrix T of shape (N, N)
        Memory storage in form of a matrix (synaptic weights)
        Its dimensions are determined by N=k*m (hash parameters)
    """
    T = np.zeros((N,N))
    # for i in range(N):
    #     line_mask = [(float(T[x,i]) if torch.rand(1) < s else 0) for x in range(N)]
    #     mask.append(line_mask)    
    if not(V is None):
        for vect in V:
            T = add_memory(T, vect, p)
            T[mask==0]=0
    return T

def add_memory(T, new_memory, p):
    """
    Eq. (2) from [1]
    
    Update synaptic weights in form of matrix T (symmetric recurrent weight matrix) when adding new memory.
    
    Parameters
    ----------
    T : np matrix T_sum of shape (N, N)
        Initialized memory storage in form of a matrix (synaptic weights)
    new_memory : np matrix of shape (1,N)
        a vector we wish to store
    p : num
        a number representing the probability of a value being 1 (i.e. sparsity)
        
    Returns
    -------
    a np matrix T of shape (N, N)
        Renewed memory storage in form of a matrix (synaptic weights)
        Its dimensions are determined by N=k*m (hash parameters)
    """
    N = T.shape[0]
#     if method == "default":
#         v = 2*new_memory - 1 #hopfield

    v = new_memory - p #tsodyks
#     elif method == "amits":
#         v = 2*new_memory - 2*p #amits
    # outer_prod = torch.outer(v,v)
    outer_prod = np.outer(v,v)
    # outer_prod.fill_diagonal_(0)
    np.fill_diagonal(outer_prod,0)
    T += outer_prod
        
    return T

# sparsify_matrix(100)
# initialize_network(10, 0.1, V=[np.array([0,1,0,1,0,0,0,0,1,0])])