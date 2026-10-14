import numpy as np

from min_quadratic import min_quadratic


def max_abs_quadratic(c, g, H, Delta):
    """
    Solve the problem max |f| on a ball centered at the origin with radius Delta,
    where
        f(x) = c + g.T@x + 0.5*x.T@H@x
    """

    # H = None --> linear function
    if H is None:
        s1, val1 = min_quadratic(g, None, Delta)
        s2, val2 = min_quadratic(-g, None, Delta)
    else:
        # maximize absolute value by minimizing (+1)* and (-1)* HIP
        s1, val1 = min_quadratic(g, H, Delta)
        s2, val2 = min_quadratic(-g, -H, Delta)

    val1 += c # add constant for f
    val2 -= c # add constant for -f

    # use formula max |f| = max(max(f), max(-f)) = max(-min(-f), -min(f))
    val = max(-val1, -val2)
    if val == -val1:
        return s1, val
    else:
        return s2, val
