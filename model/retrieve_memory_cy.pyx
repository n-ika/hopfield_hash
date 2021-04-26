import numpy as np
cimport numpy as cnp
cimport cython
from libc.stdlib cimport rand, RAND_MAX
from cpython cimport bool
ctypedef cnp.float64_t CTYPE_W # synaptic weight type
ctypedef cnp.float64_t CTYPE_A # neural activation type
ctypedef cnp.intp_t CTYPE_I # array index type
ctypedef cnp.float64_t CTYPE_float
ctypedef cnp.int_t CTYPE_int


def check_threshold(membrane_potential, U):
    """
    Check whether the sum of T_ij * V_j is bigger or smaller than the threshold U

    Parameters
    ----------
    membrane_potential : num
        Sum over T_ij * V_j
    U : num
        a scalar representing a threshold of neuron's state of activity
        
    Returns
    -------
    num
        We return either 0 or 1, depending on TV_sum being smaller
        or larger than the threshold U
    """
    if membrane_potential > U:
        return 1
    else:
        return 0


cdef _update_neuron(CTYPE_W[:,:] T, CTYPE_float U, CTYPE_A[:] V, CTYPE_I i):
    membrane_potential = np.sum(np.multiply(T[:,i],V))
    new_V_i = check_threshold(membrane_potential, U)
    return new_V_i


cdef _has_converged(CTYPE_W[:,:] T, CTYPE_float U, CTYPE_A[:] V):
    """
    Check whether the new V_i is the same as i-th value of V
    and whether the current energy is equal to the previous one

    Parameters
    ----------
    V_i : num
        state of V's i-th neuron (either active: 1, or inactive: 0)
    V : num
        a numpy array representing all current neurons' states of activity
    i : num
        current randomly chosen index
    E_list : list
        list of all energy values so far
        
    Returns
    -------
    boolean
        We return False if we satisfy at least one of the conditions,
        else we return True
    """
    cdef CTYPE_I i
    cdef bint converged = True
    # converged = True
    for i, V_i in enumerate(V):
        updated_V_i = _update_neuron(T, U, V, i)
        if updated_V_i != V_i:
            converged = False
            break
    return converged



def retrieve_memory(T, V0, U=0, SEED=27, check_frequency=1):
#     random.seed(SEED)
    conv_check_spacing = len(V0)*check_frequency 
    V = V0.copy()
    return _retrieve_memory(T.shape[0], T, V, U, conv_check_spacing)
    

cpdef _retrieve_memory(CTYPE_I N, CTYPE_W[:,:] T, CTYPE_A[:] V, CTYPE_float U, CTYPE_int conv_check_spacing):
    cdef CTYPE_I i, j
    while not _has_converged(T, U, V):
        for j in range(conv_check_spacing):
            i = 1 + int(rand()/(RAND_MAX*N)) # random integer 1,...,N
            V[i] = _update_neuron(T, U, V, i)
    return V