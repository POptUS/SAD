import numpy as np


def get_sigma(g, H, model_degree=2):

#     if model_degree == 2:
#         mineval = np.linalg.eigvalsh(H)[0]
#         sigma = max(np.linalg.norm(g), -mineval)
#     elif model_degree == 1:
#         sigma = np.linalg.norm(g)

    if H is None:
        mineval = 0.0
    else:
        mineval = np.linalg.eigvalsh(H)[0]
    sigma = max(np.linalg.norm(g), -mineval)
    
    return sigma
