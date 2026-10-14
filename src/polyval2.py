import numpy as np

from phi2eval import phi2eval, Dphi2eval
from multiindex2derivatives import multiindex2derivatives
from idx_uppertri2vec import idx_uppertri2vec

# cache dictionary shared across calls
_TRIU_CACHE = {}

def get_triu_indices(n):
    """Return cached upper-triangular indices for dimension n."""
    if n not in _TRIU_CACHE:
        iu = np.triu_indices(n)
        diag_mask = (iu[0] == iu[1])
        _TRIU_CACHE[n] = (iu, diag_mask)
    return _TRIU_CACHE[n]


def polyval2(coefs, Xeval):
    """
    evaluate quadratic polynomial from vector of coefficients in standard order
    Xeval is an m-by-n matrix of evaluation points
    """

    # read format of Xeval
    m, n = np.shape(Xeval)

    # constant term
    out = coefs[0] * np.ones(m)

    # linear terms
    out += Xeval @ coefs[1 : n + 1]

    # quadratic terms
    out += phi2eval(Xeval) @ coefs[n + 1 :]

    return out


def Dpolyval2(coefs, Xeval, alpha):
    """
    evaluate a derivative of quadratic polynomial from vector of coefficients in standard order
    Xeval is an m-by-n matrix of evaluation points
    """

    # read format of Xeval
    nd = Xeval.ndim
    if nd == 1:
        m = 1
        n = len(Xeval)
    elif nd == 2:
        m, n = np.shape(Xeval)

    # constant term
    deriv_order = sum(alpha)
    d_list = multiindex2derivatives(alpha)

    out = coefs[0] * (deriv_order == 0) * np.ones(m)

    # linear terms
    if deriv_order == 0:
        out += Xeval @ coefs[1 : n + 1]
    elif deriv_order == 1:
        j = d_list[0]
        out += coefs[1 + j] * np.ones(m)

    # quadratic terms
    if deriv_order == 0:
        out += phi2eval(Xeval) @ coefs[n + 1 :]
    elif deriv_order == 1:
        out += Dphi2eval(Xeval, j) @ coefs[n + 1 :]
    elif deriv_order == 2:
        out += coefs[n + 1 + idx_uppertri2vec(d_list[0],d_list[1],n)] * np.ones(m)

    if nd == 1:
        out = out.item() # singleton array to scalar

    return out


def Dpolyval2_combined(coefs, Xeval, alpha):
    """
    Evaluate a quadratic polynomial or its derivatives in fully vectorized form.

    coefs: [1 + n + n*(n+1)/2] coefficient vector (constant, linear, quadratic)
    Xeval: [m,n] or [n] evaluation points
    alpha: multiindex indicating derivative order (0, 1, or 2)
    """
    Xeval = np.atleast_2d(Xeval)
    m, n = Xeval.shape
    deriv_order = sum(alpha)
    d_list = multiindex2derivatives(alpha)

    # constant term
    out = np.full(m, coefs[0] * (deriv_order == 0))

    # Precompute outer products once for all quadratic features
    P = Xeval[:, :, None] * Xeval[:, None, :]     # shape (m, n, n)

	# retrieve precomputed indices and diagonal mask
    iu, diag_mask = get_triu_indices(n)

    if deriv_order == 0:
        # Linear and quadratic terms
        out += Xeval @ coefs[1:n+1]
        Phi = P[:, iu[0], iu[1]]
        Phi[:, diag_mask] *= 0.5
        out += Phi @ coefs[n+1:]

    elif deriv_order == 1:
        j = d_list[0]
        out += coefs[1+j]

        # Derivative of quadratic part wrt x_j
        # d/dx_j (x_k x_l) = δ_{jk} x_l + δ_{jl} x_k
        dPhi = np.zeros((m, iu[0].size))
        mask1 = (iu[0] == j)
        mask2 = (iu[1] == j) & (iu[0] != iu[1])  # exclude diagonal from second mask
        if np.any(mask1):
            dPhi[:, mask1] += Xeval[:, iu[1][mask1]]
        if np.any(mask2):
            dPhi[:, mask2] += Xeval[:, iu[0][mask2]]
        out += dPhi @ coefs[n+1:]

    elif deriv_order == 2:
        j1, j2 = d_list
        idx = int(n * j1 - 0.5 * j1 * (j1 + 1) + j2)  # idx_uppertri2vec(j1, j2, n)
        out += coefs[n + 1 + idx]

    if Xeval.ndim == 1:
        out = out.item()

    return out

