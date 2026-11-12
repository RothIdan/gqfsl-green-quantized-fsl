import numpy as np
from scipy.special import comb


def optimal_alpha(T, G2, L, beta, gamma, Gamma, I, qc, dc, qs, ds):
    """
    Calculates the optimal bounding parameter alpha to minimize the convergence upper bound.
    This implementation directly corresponds to Corollary 1 in the manuscript.

    Args:
        T (int): Number of global communication rounds.
        G2 (float): Estimated bounded squared norm of the stochastic gradients.
        L (float): Smoothness parameter of the loss function.
        beta (float): Learning rate scaling factor.
        gamma (float): Learning rate offset parameter.
        Gamma (float): Degree of data heterogeneity.
        I (int): Number of local iterations per global round.
        qc (int): Client-side submodel precision level in bits.
        dc (int): Number of weights in the client-side submodel.
        qs (int): Server-side submodel precision level in bits.
        ds (int): Number of weights in the server-side submodel.

    Returns:
        float: The optimal bounding parameter alpha.
    """
    mu = 2/beta

    Y = 4*beta*(I-1)**2*G2/(gamma+1) 
    Z = dc/(2**(2*qc)) + ds/(2**(2*qs))
    X = beta**2/(beta*mu-1)

    alpha_0 = np.sqrt(L*Gamma / (mu*X*Y/(2*(T*I+gamma)) + Z/2))

    return np.max([alpha_0, L, L*(1+gamma)/(1+gamma-L*beta)])



def convergence_bound(T, G2, sigma2, L, mu, beta, Gamma, p, gamma, dc, ds, qc, qs, qu, I, K, N):
    """
    Calculates the theoretical upper bound of the expected optimality gap.
    This function implements Theorem 1 from the manuscript.

    Args:
        T (int): Number of global communication rounds.
        G2 (float): Estimated bounded squared norm of the stochastic gradients.
        sigma2 (float): Estimated variance of the stochastic gradients.
        L (float): Smoothness parameter of the loss function.
        mu (float): Strong convexity parameter of the loss function.
        beta (float): Learning rate scaling factor.
        Gamma (float): Degree of data heterogeneity.
        p (array): Sampling probability for all clients.
        gamma (float): Learning rate offset parameter.
        dc (int): Number of weights in the client-side submodel.
        ds (int): Number of weights in the server-side submodel.
        qc (int): Client-side submodel precision level in bits.
        qs (int): Server-side submodel precision level in bits.
        qu (int): Transmission precision level in bits for model uploads.
        I (int): Number of local iterations per global round.
        K (int): Number of active clients participating in the round.
        N (int): Total number of clients in the network.

    Returns:
        float: The theoretical upper bound of the loss function gap.
    """
    
    alpha = optimal_alpha(T, G2, L, beta, gamma, Gamma, I, qc, dc, qs, ds)
    convergence_bound.alpha = alpha # Store the optimal alpha for later tracking and analysis

    psi_1 = (alpha - mu) * (dc/(2**(2*qc)) + ds/(2**(2*qs))) + 2*L*Gamma/alpha
    psi_2 = np.sum(p**2 * sigma2) + (4*alpha*beta*(I-1)**2*G2)/(gamma+1) + 2*L*Gamma + (4*dc*I*G2)/(K*2**(2*qu)) + 4*I**2*G2*(N-K)/(K*(N-1)) 

    h = ( L/(2*(T*I+gamma)) ) * ( (beta**2*psi_2)/(beta*mu-1) + (gamma+1)*G2/mu**2 ) + (L*psi_1)/(2*mu)

    return h



def find_minimal_T(a, b, constraint_func, epsilon, *args):
    """
    Determines the minimal number of global rounds T required to satisfy the target accuracy.
    Uses a discrete bisection search over a defined range.

    Args:
        a (int): Lower bound of the search range.
        b (int): Upper bound of the search range.
        constraint_func (callable): The function calculating the convergence bound.
        epsilon (float): The target accuracy threshold constraint.
        *args: Fixed framework parameters passed to the constraint function.

    Returns:
        tuple: A tuple containing the minimal required rounds (int) and the optimal alpha (float). 
               Returns (None, None) if the constraint cannot be met within the range.
    """

    low = a
    high = b
    ans = b

    # Check the boundary first to ensure a solution exists in the range
    if not constraint_func(b, *args) <= epsilon:
        return None, None # Constraint not met within [a, b]

    alpha = constraint_func.alpha

    while (high - low) > 0:
        mid = (low + high) // 2
            
        # Call function with mid as T and unpacking the rest of the params
        if constraint_func(mid, *args) <= epsilon:
            ans = mid
            high = mid - 1 
            alpha = constraint_func.alpha
        else:
            low = mid + 1 
            
        if low > high:
            break
            
    return ans, alpha




def expected_max_subset(values, K):
    """
    Calculates the expected maximum value of a randomly drawn subset without replacement.
    This function implements Lemma 2 from the manuscript to analytically 
    evaluate the expected per-round straggler latency and broadcasting energy.

    Args:
        values (array-like): The full population array of size N.
        K (int): The number of elements drawn in the subset.

    Returns:
        float: The expected maximum value of the sampled subset.
        
    Raises:
        ValueError: If K is less than or equal to 0 or greater than N.
    """
    
    values = np.array(values)
    N = len(values)
    
    # Edge case checks
    if K <= 0 or K > N:
        raise ValueError(f"K={K} must be between 1 and N={N}.")
    if K == 1:
        return np.mean(values) # If K=1, the expected max is just the population mean
        

    v_sorted = np.sort(values) # Sort the array in ascending order
    

    i = np.arange(K, N + 1)
    
    ways_to_be_max = comb(i - 1, K - 1, exact=False) # Array of i-1 chose K-1 values for all relevant values in i (i = K,...,N)
    total_ways = comb(N, K, exact=False)
    
    probabilities = ways_to_be_max / total_ways
    
    # Multiply the valid sorted values by their probabilities and sum
    expected_max = np.sum(v_sorted[K - 1:] * probabilities)
    
    return expected_max
