import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from get_poisedness import get_poisedness

sys.path.append("../benchmark")
from make_alpha_admissible_from_K import make_alpha_admissible_from_K


def heatmap_poisedness(
    Y,
    K,
    ykin,
    Delta,
    model_degree=2,
    save_dir="plots",
    filename="heatmap_poisedness.png",
    Data=None,
):
    """
    Y [dbl] [npts-by-n] points held constant when determining poisedness of the interpolation set
            must have n = 2
    K [int] indices of known partial derivatives of the objective function
    ykin [int] index of model center point
    Delta [dbl] trust region radius
    Data [list] interpolation conditions of form (point_index, alpha)
    """

    n = len(Y[0])
    alpha_admissible = make_alpha_admissible_from_K(K, n, max_deriv=1)

    # Discretize the trust region in polar coordinates
    n_angle = 1001
    n_r = 100

    angle = np.linspace(0, 2 * np.pi, n_angle)
    r = np.linspace(Delta / n_r, Delta, n_r)
    R, ANGLE = np.meshgrid(r, angle)

    delta_sphere = Delta * np.array([np.cos(angle), np.sin(angle)]).T + Y[ykin, :]

    # Coordinates of point to vary
    X0_mat = R * np.cos(ANGLE) + Y[ykin, 0]
    X1_mat = R * np.sin(ANGLE) + Y[ykin, 1]

    # Default: use all alpha_admissible data at all fixed points and the added point
    if Data is None:
        Data = []
        for i in range(len(Y) + 1):
            for alpha in alpha_admissible:
                Data.append((i, alpha))

    # Test poisedness of each additional point to Y
    LAMBDA = np.empty((n_angle, n_r))
    LAMBDA.fill(np.nan)

    for i in range(n_angle):
        print(f"angle {i}")
        for j in range(n_r):
            newpoint = np.array([X0_mat[i, j], X1_mat[i, j]])
            Ynew = np.concatenate((Y, [newpoint]), axis=0)

            LAMBDA[i, j] = get_poisedness(
                Ynew,
                alpha_admissible,
                ykin,
                Data,
                Delta,
                model_degree=model_degree,
            )

            if LAMBDA[i, j] > 1e10:
                LAMBDA[i, j] = np.inf

    # Plot
    fig, ax = plt.subplots()

    surface = ax.pcolormesh(
        X0_mat,
        X1_mat,
        np.log(LAMBDA),
        cmap=cm.viridis,
        vmin=1,
        vmax=4,
    )

    ax.plot(delta_sphere[:, 0], delta_sphere[:, 1], "k--")

    # Create a dictionary mapping each index to its set of usages
    usage = defaultdict(set)
    for idx, role in Data:
        usage[idx].add(role)

    # Create legend handles
    legend_handles = {}

    # Plot each fixed point with appropriate marker
    for i in range(len(Y)):
        roles = usage[i]
        if (0, 0) in roles and (1, 0) in roles:
            marker = "s"  # square
            label = "f and df"
        elif (0, 0) in roles:
            marker = "o"  # circle
            label = "f only"
        elif (1, 0) in roles:
            marker = "^"  # triangle
            label = "df only"
        else:
            marker = "x"  # unused
            label = "Unused"

        h = ax.plot(
            Y[i, 0],
            Y[i, 1],
            marker,
            markersize=10.0,
            color="white",
            markeredgecolor="black",
            markeredgewidth=1.5,
            label=label,
        )

        # Save handle for legend, but only once per label
        if label not in legend_handles:
            legend_handles[label] = h[0]

    if legend_handles:
        ax.legend(legend_handles.values(), legend_handles.keys(), loc="upper left", frameon=True)

    cbar = fig.colorbar(surface, ax=ax)
    cbar.set_ticks([1, 2, 3, 4])
    cbar.set_ticklabels([r"$10^1$", r"$10^2$", r"$10^3$", r"$10^4$"])

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")

    fac = 1.05
    ax.axis(
        [
            Y[ykin, 0] - fac * Delta,
            Y[ykin, 0] + fac * Delta,
            Y[ykin, 1] - fac * Delta,
            Y[ykin, 1] + fac * Delta,
        ]
    )

    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Best found poisedness constant: Lambda = %1.3e" % (LAMBDA.min()))


#######################################################
if __name__ == "__main__":
    K = np.array([0])
    ykin = 0
    Delta = 1.0

    Y = np.array([[0.0, 0.0], [0.0, 1.0]])
    heatmap_poisedness(
        Y,
        K,
        ykin,
        Delta,
        model_degree=2,
        save_dir="plots",
        filename="heatmap_poisedness_axis.png",
    )

    Y = np.array([[0.0, 0.0], [np.sqrt(2) / 2, np.sqrt(2) / 2]])
    heatmap_poisedness(
        Y,
        K,
        ykin,
        Delta,
        model_degree=2,
        save_dir="plots",
        filename="heatmap_poisedness_diag.png",
    )

    Y = np.array(
        [
            [0.0, 0.0],
            [np.sqrt(2) / 2, np.sqrt(2) / 2],
            [np.sqrt(2) / 2, -np.sqrt(2) / 2],
        ]
    )

    Data = [
        (0, (0, 0)),
        (0, (1, 0)),
        (1, (0, 0)),
        (1, (1, 0)),
        (2, (0, 0)),
        (3, (1, 0)),
    ]

    heatmap_poisedness(
        Y,
        K,
        ykin,
        Delta,
        model_degree=2,
        save_dir="plots",
        filename="heatmap_poisedness_diag_f_only_plus_df_newpoint.png",
        Data=Data,
    )

    Data = [
        (0, (0, 0)),
        (0, (1, 0)),
        (1, (0, 0)),
        (1, (1, 0)),
        (2, (1, 0)),
        (3, (0, 0)),
    ]

    heatmap_poisedness(
        Y,
        K,
        ykin,
        Delta,
        model_degree=2,
        save_dir="plots",
        filename="heatmap_poisedness_diag_df_only_plus_f_newpoint.png",
        Data=Data,
    )

    Data = [
        (0, (0, 0)),
        (0, (1, 0)),
        (1, (0, 0)),
        (1, (1, 0)),
        (2, (0, 0)),
        (3, (0, 0)),
    ]

    heatmap_poisedness(
        Y,
        K,
        ykin,
        Delta,
        model_degree=2,
        save_dir="plots",
        filename="heatmap_poisedness_diag_f_only_plus_f_newpoint.png",
        Data=Data,
    )
