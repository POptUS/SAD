import numpy as np
from itertools import combinations_with_replacement

def make_alpha_admissible_from_K(K, n, max_deriv=2):
    """
    Construct all admissible derivative multi-indices, assuming all orders
    of derivatives up to max_deriv are known with respect to variables in K.
    K [<=n] is a numpy array
    n [int] input dimension
    """
    
    alpha_admissible = []

    # For each possible degree from 0 to max_deriv
    for total_deg in range(max_deriv + 1):

        # Generate possible alphas of degree total_deg
        for partial_alpha in combinations_with_replacement(range(len(K)), total_deg):
            counts = [0] * len(K)
            for idx in partial_alpha:
                counts[idx] += 1
            
            # Create the full alpha vector of length n
            full_alpha = [0] * n
            for i, k_idx in enumerate(K):
                full_alpha[k_idx] = counts[i]
            
            alpha_admissible.append(tuple(full_alpha))

    return alpha_admissible


if __name__ == "__main__":
    
    n = 2
    K = np.array([0], dtype=int)
    max_deriv = 1
    alpha_admissible = make_alpha_admissible_from_K(K, n, max_deriv)
    print(alpha_admissible)
    
    n = 2
    K = np.array([], dtype=int)
    max_deriv = 1
    alpha_admissible = make_alpha_admissible_from_K(K, n, max_deriv)
    print(alpha_admissible)

    n = 5
    K = np.array([0, 2, 4], dtype=int)
    max_deriv = 2
    alpha_admissible = make_alpha_admissible_from_K(K, n, max_deriv)
    print(alpha_admissible)
