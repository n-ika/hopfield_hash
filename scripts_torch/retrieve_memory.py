import random
import torch

def retrieve_memory(T, V0, U=0, downstate=0, full_trace=False, SEED=27, check_frequency=1):
    """
    Eq. (1) from [1]
    
    To retrieve a memory, we want to find the stable/fixed point of the 
    dynamic network represented by matrix T (synaptic weights in which
    the memory is stored) when starting from vector V.

    Parameters
    ----------
    T : a torch tensor T of shape (N, N)
        Memory in form of a matrix (synaptic weights)
    V0 : a torch tensor of shape (1, N)
        a vector with which we initialize the network activity (check if it is stored in T)
    g : num
        parameter
    U : num
        a scalar representing the threshold of neuron's state of activity.
        Set to 0 by default.
        "With a threshold of 0, the system behaves as a forced categorizer." [1]
    full_trace : boolean
        Set to True by default. This means we will keep all the changes of the initial neuron states
        as they change through time.
    SEED : num
        Used for the random choices of indices, which we can control for replication by always 
        setting the seed to the same number.
        
    Returns
    -------
    a torch tensor of shape (1, N)
        We return the new / denoised V
    """
    torch.manual_seed(SEED)
    conv_check_spacing = len(V0)*check_frequency 
    
    V = V0.detach().clone()
    if full_trace:
        V_history = [V.detach().clone()]
    k = 0
    while not has_converged(T, U, V, downstate):
        for _ in range(conv_check_spacing):
            idx = torch.randint(len(V), (int(V.shape[0]*0.1),)) # 10% of idxs, random choice w/ replacement
            # print(idx)
            # i = random.randrange(V.shape[0])
            V[idx] = update_neuron(T, U, V, idx, downstate)
            k += 1
            if full_trace:
                V_history.append(V.detach().clone())

    if full_trace:
        return V_history, k
    else:
        return V, k


def update_neuron(T, U, V, idx, downstate=0):
    """
    We calculate sum_j {T_ji * V[j]}, where T_ij is our synaptic weight between neuron j and i
    and V[j] is a j-th neuron state 
    
    Parameters
    ----------
    T : a numpy array T_sum of shape (N, N)
        Initialized memory storage in form of a matrix (synaptic weights)
    V : num
        a scalar representing the neural network state
        
    Returns
    -------
    num
        We return the sum of all components of i-th column of T, each multiplied by V_j
    """
    new_V_i = []
    for i in idx:
        membrane_potential = torch.sum(torch.mul(T[:,i.item()],V))
        new_V_i.append(check_threshold(membrane_potential, U, downstate))
    return torch.tensor(new_V_i).long()



def check_threshold(membrane_potential, U, downstate):
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
    
    
def has_converged(T, U, V, downstate):
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
    converged = True
    for i, V_i in enumerate(V):
        updated_V_i = update_neuron(T, U, V, torch.tensor([i]), downstate)
        if updated_V_i.all() != V_i.all():
            converged = False
            break
    return converged 