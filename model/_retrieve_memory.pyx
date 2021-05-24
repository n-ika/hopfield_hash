import numpy as np
cimport numpy as cnp
cimport cython
#from libc.stdlib cimport rand, RAND_MAX
import numpy.random as random
ctypedef cnp.float64_t CTYPE_W # synaptic weight type
ctypedef cnp.float64_t CTYPE_A # neural activation type
ctypedef Py_ssize_t CTYPE_I # array index type
ctypedef cnp.int_t CTYPE_S # neuron activation
ctypedef cnp.int_t CTYPE_int
ctypedef cnp.float64_t CTYPE_float

def make_table(N, SEED=27):
    np.random.seed(SEED)
    return np.random.randint(0,N,size=(1000000,))

def retrieve_memory(T, V0, U=0.0, SEED=27, check_frequency=1):
    random.seed(SEED)
    # create array of random numbers here
    conv_check_spacing = len(V0)*check_frequency 
    V = V0.copy()
    return _retrieve_memory(T.shape[0], T, V.astype(int), U, conv_check_spacing)
    

@cython.cdivision(True)
cdef _retrieve_memory(CTYPE_I N, CTYPE_W[:,:] T, CTYPE_S[:] V, CTYPE_A U, CTYPE_int conv_check_spacing):
    cdef CTYPE_I i, j, k
    cdef CTYPE_int[:] random_table
    random_table = make_table(N)
    k = 0
    while not _has_converged(N, T, U, V):
        for j in range(conv_check_spacing):
            i =  random_table[k,] #(rand()*N)/RAND_MAX # random integer 0,...,N-1
            V[i] = _update_neuron(N, T, U, V, i)
            k += 1
            if k >= random_table.shape[0]:
                print("Re-making table, k=",k)
                _retrieve_memory(N, T, V, U, conv_check_spacing)
                # reprovision table and reset correct running context
    
    with open(str(N)+"_k.txt", "a+") as file_object:
        file_object.write(str(k) + "\n")
    return V


cdef _check_threshold(CTYPE_A membrane_potential, CTYPE_A U):
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


cdef _update_neuron(CTYPE_I N, CTYPE_W[:,:] T, CTYPE_A U, CTYPE_S[:] V, CTYPE_I i):
    cdef CTYPE_I j
    cdef CTYPE_A membrane_potential = 0.
    for j in range(N):
        membrane_potential += T[j,i]*V[j]
    new_V_i = _check_threshold(membrane_potential, U)
    return new_V_i


cdef _has_converged(CTYPE_I N, CTYPE_W[:,:] T, CTYPE_A U, CTYPE_S[:] V):
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
    cdef bint updated_V_i
    cdef bint converged = True
    # converged = True
    for i in range(N):
        updated_V_i = _update_neuron(N, T, U, V, i)
        if updated_V_i != V[i]:
            converged = False
            break
    return converged

