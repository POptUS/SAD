import numpy as np
import trustregion
from sys import exit
try:
    from minqsw import minqsw
except ModuleNotFoundError as e:
    print(e)
    exit("Ensure a python implementation of MINQ is available. For example, clone https://github.com/POptUS/minq and add minq/py/minq5 to the PYTHONPATH environment variable")

def min_quadratic(g, H, Delta):
    """
    minimize the quadratic function given by
        f(x) = g.T@x + 0.5*x.T@H@x
    over a ball centered at the origin with radius Delta
    """

    n = len(g)

    if np.linalg.norm(g) < 1e-14:  # NOTE: need to perturb g if gradient at initial point (0) exactly vanishes
        g_temp = 1e-14 * np.ones(n)
        x = trustregion.solve(g_temp, H, Delta)
    else:
        x = trustregion.solve(g, H, Delta)

    if H is None:
        minval = x.dot(g)
    else:
        minval = x.dot(g + 0.5 * H.dot(x))

    rng = np.random.default_rng(seed=44444)
    npts = 50
    rads = rng.uniform(0, Delta, npts)**(1/n)
    X0 = rng.normal(size=(npts, n)) # generate from n-dim Gaussian
    X0 /= np.linalg.norm(X0, axis=1, keepdims=True) # rescale to unit sphere
    X0 *= rads[:, None] # random radius

    for i in range(npts): # sample 

        xtry = X0[i]
        val = xtry.dot(0.5 * H.dot(xtry) + g)  # model at candidate point

        if val < minval:
            x = xtry
            minval = val

    # attempt to minimize further with minq from best x so far
    [Xsp, _, minq_err, _] = minqsw(0, g, H, -Delta*np.ones(n), Delta*np.ones(n), 0, np.reshape(x, (n,1))) #np.zeros((n, 1)))
    Xsp = Xsp.squeeze()
    x_minq = Delta*Xsp/np.linalg.norm(Xsp) # Box-norm solution projected to two-norm sphere

    val_minq = x_minq.dot(0.5 * H.dot(x_minq) + g)  # model at candidate point

    if val_minq < minval:
        x = x_minq
        minval = val_minq

    return x, minval
