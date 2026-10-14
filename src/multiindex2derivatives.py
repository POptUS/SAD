from functools import lru_cache
import numpy as np

@lru_cache(maxsize=None)
def multiindex2derivatives_cached(alpha_tuple):
    alpha = np.fromiter(alpha_tuple, dtype=int)
    return np.repeat(np.arange(alpha.size), alpha)

def multiindex2derivatives(alpha):
    """
    Convert a multi-index alpha into a list of variable indices for differentiation.

    Input:
    alpha: A list or numpy array of integers representing the multi-index (e.g., [1, 0, 1] for 3 variables).

    Output:
    A list of indices (with repeats) indicating the variables with respect to which differentiation should occur.
    """
    # derivative_indices = []
    # for j, count in enumerate(alpha):
    #     derivative_indices.extend([j] * count)  # Add j to the list 'count' times
    # return derivative_indices
    return multiindex2derivatives_cached(tuple(alpha))

# # Example usage:
# alpha = [1, 0, 1]  # Differentiate once with respect to the 0th variable, once with respect to the 2nd variable
# result = multiindex_to_derivatives(alpha)
# print(result)  # Output: [0, 2]
