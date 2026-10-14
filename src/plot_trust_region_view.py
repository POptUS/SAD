import matplotlib.pyplot as plt
import numpy as np


def plot_trust_region_view(X, F, xkin, Data, Delta, fun, it, xtry=None, model_quad=None, save_dir="plots"):
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # plot 2 variables in a plane
    # the fixed variables are held at their vales in X[xkin]
    vary_ax = np.array([0, 1], dtype=int)
    _, n = np.shape(X)
    fixed_ax = np.setdiff1d(range(n), vary_ax)

    # EVAN note: SENSORS: 3.0; HILBERTA: 4.0; MODBEALE: 4.0; ENGVAL1: 4.0
    bnd = 4.0 # bounds for plot window = +-bnd

    os.makedirs(save_dir, exist_ok=True)

    def get_background(ax):
        # Grid for contours
        xx, yy = np.meshgrid(np.linspace(-bnd, bnd, 300), np.linspace(-bnd, bnd, 300))
        ZZ = np.zeros_like(xx)
        xeval = X[xkin].copy()
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                xeval[vary_ax[0]] = xx[i,j]
                xeval[vary_ax[1]] = yy[i,j]
                ZZ[i, j] = fun(xeval)[0]

        levels = np.concatenate([
            np.logspace(0, 2.5, 20),
        ])
        # EVAN note: use the following levels instead for SENSORS, since f takes negative values
        # levels = np.concatenate([
        #     np.linspace(-1.0, 1.0, 20),
        # ])
        ax.contour(xx, yy, ZZ, levels=levels, cmap='coolwarm', alpha=0.8, linewidths=1)
        return xx, yy

    def add_standard_elements(ax):
        # Points used in model
        Xmodel_idx = sorted(set(i for i, _ in Data))
        roles = [tuple(a) for (_, a) in Data]
        used_f = [i for (i, a) in Data if sum(a) == 0]
        used_df = [i for (i, a) in Data if sum(a) == 1] # EVAN: changed for n > 2
        used_d2f = [i for (i, a) in Data if sum(a) == 2]

        # used_both = list(set(used_f).intersection(set(used_df)))
        # only_f = list(set(used_f) - set(used_both))
        # only_df = list(set(used_df) - set(used_both))

        used_all = list(set(used_f).intersection(set(used_df)).intersection(used_d2f))
        used_f_df = list(set(used_f).intersection(set(used_df)) - set(used_all))
        used_f_d2f = list(set(used_f).intersection(set(used_d2f)) - set(used_all))
        used_df_d2f = list(set(used_df).intersection(set(used_d2f)) - set(used_all))
        only_f = list(set(used_f) - set(used_all) - set(used_f_df) - set(used_f_d2f))
        only_df = list(set(used_df) - set(used_all) - set(used_f_df) - set(used_df_d2f))
        only_d2f = list(set(used_d2f) - set(used_all) - set(used_f_d2f) - set(used_df_d2f))

        if only_f:
            ax.plot(X[only_f, vary_ax[0]], X[only_f, vary_ax[1]], 'o', label="f only", color='blue')
        if only_df:
            ax.plot(X[only_df, vary_ax[0]], X[only_df, vary_ax[1]], '^', label="df only", color='green')
        if only_d2f:
            ax.plot(X[only_d2f, vary_ax[0]], X[only_d2f, vary_ax[1]], 's', label="d2f only", color='orange')
        if used_f_df:
            ax.plot(X[used_f_df, vary_ax[0]], X[used_f_df, vary_ax[1]], 'P', label="f & df", color='black')
        if used_f_d2f:
            ax.plot(X[used_f_d2f, vary_ax[0]], X[used_f_d2f, vary_ax[1]], '*', label="f & d2f", color='cyan')
        if used_df_d2f:
            ax.plot(X[used_df_d2f, vary_ax[0]], X[used_df_d2f, vary_ax[1]], 'x', label="df & d2f", color='yellow')
        if used_all:
            ax.plot(X[used_all, vary_ax[0]], X[used_all, vary_ax[1]], '1', label="used all", color='brown')

        # Current xk
        ax.plot(X[xkin, vary_ax[0]], X[xkin, vary_ax[1]], 'ro', label="x_k", markersize=8)

        # Trust region
        tr_circle = plt.Circle(X[xkin, vary_ax], Delta, color='r', fill=False, linestyle='--', linewidth=2.5, label='Trust region')
        ax.add_patch(tr_circle)

        ax.set_title(f"Major Iteration {it}")
        ax.axis('equal')
        ax.set_xlim([-bnd, bnd])
        ax.set_ylim([-bnd, bnd]) # [-0.5, 1.5]
        ax.legend()

    def save_before():
        fig, ax = plt.subplots()
        get_background(ax)
        add_standard_elements(ax)
        fig.savefig(os.path.join(save_dir, f"iter_{it:03d}_1_before.png"), bbox_inches='tight', dpi=300)
        plt.close(fig)

    def save_after():
        fig, ax = plt.subplots()
        get_background(ax)
        add_standard_elements(ax)
        if xtry is not None:
            ax.plot(xtry[vary_ax[0]], xtry[vary_ax[1]], 'md', label="x_try", markersize=10)
        fig.savefig(os.path.join(save_dir, f"iter_{it:03d}_2_after.png"), bbox_inches='tight', dpi=300)
        plt.close(fig)

    def save_model_contour():
        if xtry is None or model_quad is None:
            return

        fig, ax = plt.subplots()
        xx, yy = get_background(ax)
        add_standard_elements(ax)

        g, H = model_quad
        Zmodel = np.full_like(xx, np.nan)
        center = X[xkin]
        pt = center.copy()
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                pt[vary_ax[0]] = xx[i,j]
                pt[vary_ax[1]] = yy[i,j]
                # pt = np.array([xx[i, j], yy[i, j]])
                if np.linalg.norm(pt - center) <= Delta:
                    d = pt - center
                    Zmodel[i, j] = F[xkin] + g @ d + 0.5 * d @ H @ d

        ax.contour(xx, yy, Zmodel, levels=20, cmap='viridis', alpha=0.9, linewidths=2.5)
        fig.savefig(os.path.join(save_dir, f"iter_{it:03d}_3_model.png"), bbox_inches='tight', dpi=300)
        plt.close(fig)

    # Call the individual plot steps
    save_before()
    save_after()
    save_model_contour()

