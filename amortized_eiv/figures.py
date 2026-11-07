"""Plotting for the paper-style examples."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import metrics as M

COLORS = ["#d62728", "#f2a900", "#1f9bde"]


def _break_wraps(a):
    """Insert NaN where an angle trace wraps, so no line is drawn across the plot."""
    a = np.asarray(a, dtype=float).copy()
    a[1:][np.abs(np.diff(a)) > np.pi] = np.nan
    return a


def plot_fig2(res, path):
    ring, x, y = res["ring"], res["x"], res["y"]
    K = res["kappas"]
    fig = plt.figure(figsize=(15, 8.5))
    gs = fig.add_gridspec(3, 5, width_ratios=[1.3, 1.5, 1.0, 1.0, 1.1], hspace=0.45, wspace=0.38)
    t = np.arange(150)
    for r, k in enumerate(K):
        c = COLORS[r]
        b = res["b"][r]; f = res["free"][r]
        conv = M.kernel_tuning(y, b, ring)
        # A: time series
        ax = fig.add_subplot(gs[r, 0])
        ax.plot(t, _break_wraps(b[t]), "k", lw=0.8, label="observed behavior")
        ax.plot(t, _break_wraps(x[t]), color=c, lw=1.4, label="true latent")
        ax.set_ylim(-np.pi, np.pi); ax.set_yticks([-np.pi, 0, np.pi]); ax.set_yticklabels(["-π", "0", "π"])
        ax.set_ylabel(f"κ = {k:g}\nangle"); ax.set_xlabel("time bin")
        if r == 0:
            ax.set_title("A  behavior and latent", loc="left"); ax.legend(fontsize=7, loc="upper right")
        # B: tuning curves for two neurons
        sub = gs[r, 1].subgridspec(2, 2, hspace=0.5, wspace=0.35)
        order = np.argsort(-res["true"].std(0))[:2]
        for q, n in enumerate(order):
            for col, (curve, name, ls) in enumerate(((conv[:, n], "conventional", "k"),
                                                      (f["est"][:, n], "EIV estimate", ":"))):
                ax = fig.add_subplot(sub[q, col])
                ax.fill_between(ring, res["true"][:, n], color=c, alpha=0.3, lw=0)
                ax.plot(ring, curve, color="k" if col == 0 else c, ls="-" if col == 0 else ":", lw=1.4)
                ax.set_xticks([-np.pi, 0, np.pi]); ax.set_xticklabels(["-π", "0", "π"], fontsize=7)
                ax.tick_params(labelsize=7)
                if r == 0 and q == 0:
                    ax.set_title(name, fontsize=8)
        fig.text(0.0, 0.0, "")
        # C: reconstruction vs target
        ax = fig.add_subplot(gs[r, 2])
        ax.scatter(x, b, s=2, color="k", alpha=0.35)
        ax.scatter(x, f["mu"], s=2, color=c, alpha=0.7)
        ax.plot([-np.pi, np.pi], [-np.pi, np.pi], color="gray", lw=0.6)
        ax.set_xlabel("target latent"); ax.set_ylabel("recon. (color) / observed (black)")
        ax.set_xticks([-np.pi, 0, np.pi]); ax.set_xticklabels(["-π", "0", "π"])
        ax.set_yticks([-np.pi, 0, np.pi]); ax.set_yticklabels(["-π", "0", "π"])
        if r == 0: ax.set_title("C  latent recovery", loc="left")
        # D: error distributions
        ax = fig.add_subplot(gs[r, 3])
        bins = np.linspace(-np.pi, np.pi, 41)
        e_true = np.angle(np.exp(1j * (x - b)))
        e_hat = np.angle(np.exp(1j * (f["mu"] - b)))
        ax.hist(e_true, bins=bins, density=True, color="gray", alpha=0.6, label="x_true − s")
        ax.hist(e_hat, bins=bins, density=True, color=c, alpha=0.6, label="x̂ − s")
        ax.set_xticks([-np.pi, 0, np.pi]); ax.set_xticklabels(["-π", "0", "π"]); ax.set_xlabel("error")
        if r == 0: ax.set_title("D  representation error", loc="left"); ax.legend(fontsize=7)
        # E: held-out likelihood vs kappa
        ax = fig.add_subplot(gs[r, 4])
        cv = res["cv"][r]; g = res["grid"]
        mean, sd = cv.mean(1), cv.std(1)
        ax.plot(g, mean, color=c); ax.fill_between(g, mean - sd, mean + sd, color=c, alpha=0.3)
        ax.axvline(k, color="k", lw=1, label="true κ")
        ax.axvline(f["kappa"], color="k", lw=1, ls="--", label="learned κ")
        ax.set_xscale("log"); ax.set_xlabel("κ (fixed)"); ax.set_ylabel("held-out log p(y|b) / bin")
        if r == 0: ax.set_title("E  held-out likelihood", loc="left"); ax.legend(fontsize=7)
    fig.suptitle("Reproduction of Fig 2 (simulated, N = 30, T = 1000)", y=0.995)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_fig1(res, path):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    ring, x, y = res["ring"], res["x"], res["y"]
    titles = ["aligned", "aligned but noisy", "uncorrelated"]
    tr = res["true"]
    top = np.argsort(-tr.std(0))[:3]
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.38, wspace=0.28, height_ratios=[1, 1, 1.5])
    t = np.arange(300)
    for c, k in enumerate(res["kappas"]):
        b = res["b"][c]; f = res["fits"][c]
        conv = M.kernel_tuning(y, b, ring)
        ax = fig.add_subplot(gs[0, c])
        ax.scatter(t, b[t], s=5, color="gray", alpha=0.7, label="observed s")
        ax.plot(t, _break_wraps(x[t]), "k", lw=1.2, label="latent x")
        ax.set_ylim(-np.pi, np.pi); ax.set_yticks([-np.pi, 0, np.pi]); ax.set_yticklabels(["-π", "0", "π"])
        ax.set_title(f"{'ABC'[c]}  {titles[c]} (κ = {k:g})", loc="left"); ax.set_xlabel("time bin")
        if c == 0: ax.legend(fontsize=8, loc="lower left")
        ax = fig.add_subplot(gs[1, c]); n = top[0]
        ax.scatter(b, y[:, n], s=4, color="gray", alpha=0.4, label="spike counts vs s")
        ax.plot(ring, conv[:, n], color="#d62728", lw=2, label="conventional")
        ax.plot(ring, tr[:, n], color="k", lw=1.6, label="latent tuning (truth)")
        ax.plot(ring, f["est"][:, n], color="#1f9bde", lw=1.6, ls=":", label="EIV estimate")
        ax.set_xticks([-np.pi, 0, np.pi]); ax.set_xticklabels(["-π", "0", "π"])
        ax.set_xlabel("orientation"); ax.set_ylabel("firing rate")
        ax.set_title(f"{'DEF'[c]}  example neuron", loc="left")
        if c == 0: ax.legend(fontsize=7)
        ax = fig.add_subplot(gs[2, c], projection="3d")
        ax.scatter(*y[:, top].T, c=b, cmap="hsv", vmin=-np.pi, vmax=np.pi, s=5, alpha=0.5)
        ax.plot(*tr[:, top].T, color="k", lw=2)
        ax.plot(*conv[:, top].T, color="#d62728", lw=2)
        ax.plot(*f["est"][:, top].T, color="#1f9bde", lw=1.5, ls=":")
        ax.set_xlabel("neuron 1", fontsize=8); ax.set_ylabel("neuron 2", fontsize=8); ax.set_zlabel("neuron 3", fontsize=8)
        ax.set_title(f"{'GHI'[c]}  population manifold", loc="left")
    fig.suptitle("Reproduction of Fig 1 (simulated). Spikes and latent are identical across columns", y=0.96)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
