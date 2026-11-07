"""Build all README figures into figures/. Reads saved results; the noisy-behavior
figure refits one small model (about 40 s)."""
import json
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

OUT = "figures"
os.makedirs(OUT, exist_ok=True)
C = {"amort": "#1f5fa8", "amort2": "#7fb2e5", "qmc": "#e08a1e", "grid": "#2e9e5b",
     "lbfgs": "#8a4fb0", "truth": "#222222", "red": "#d62728"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 140, "savefig.bbox": "tight"})


# ------------------------------------------------------------------ schematic

def box(ax, xy, w, h, text, fc="#eef3fa", ec="#1f5fa8", fs=10, lw=1.4):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fc, ec=ec, lw=lw))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs)


def node(ax, xy, r, text, fc="white", ec="#222222", fs=11):
    ax.add_patch(Circle(xy, r, fc=fc, ec=ec, lw=1.5))
    ax.text(xy[0], xy[1], text, ha="center", va="center", fontsize=fs)


def arrow(ax, a, b, text=None, off=(0, 0), color="#444444", fs=9, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13, lw=1.4,
                                 color=color, connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((a[0] + b[0]) / 2 + off[0], (a[1] + b[1]) / 2 + off[1], text, fontsize=fs,
                ha="center", va="center", color=color)


def fig_model():
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axs[0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis("off")
    ax.set_title("Generative model", loc="left", fontweight="bold")
    box(ax, (3.2, 6.6), 3.6, 0.9, "learned prior  p(x)", fs=10)
    node(ax, (5.0, 4.9), 0.62, "x$_t$")
    arrow(ax, (5.0, 6.55), (5.0, 5.55))
    node(ax, (1.9, 2.5), 0.62, "b$_t$", fc="#e9e9e9")
    node(ax, (8.1, 2.5), 0.62, "y$_{nt}$", fc="#e9e9e9")
    arrow(ax, (4.5, 4.45), (2.3, 2.95), "von Mises\nwith fidelity κ$_t$", off=(-1.35, 0.35))
    arrow(ax, (5.5, 4.45), (7.7, 2.95), "Poisson with\ntuning f$_n$(x)", off=(1.35, 0.35))
    ax.text(1.9, 1.35, "measured behavior\n(noisy, may decouple)", ha="center", fontsize=9, color="#555555")
    ax.text(8.1, 1.35, "spike counts\n(neuron n, bin t)", ha="center", fontsize=9, color="#555555")
    ax.text(5.0, 0.25, "x is the internal representation. κ says how tightly it tracks behavior.",
            ha="center", fontsize=9.5, style="italic")

    ax = axs[1]
    ax.set_xlim(0, 12); ax.set_ylim(0, 8); ax.axis("off")
    ax.set_title("Encoder with a behavior gate", loc="left", fontweight="bold")
    node(ax, (1.0, 5.6), 0.55, "y$_t$", fc="#e9e9e9")
    box(ax, (2.2, 5.1), 2.7, 1.0, "network\nz$_ψ$(y$_t$)", fs=10)
    arrow(ax, (1.55, 5.6), (2.2, 5.6))
    node(ax, (7.3, 5.6), 0.45, "+", fs=14)
    arrow(ax, (4.9, 5.6), (6.85, 5.6))
    node(ax, (1.0, 2.4), 0.55, "b$_t$", fc="#e9e9e9")
    box(ax, (2.2, 1.9), 2.7, 1.0, "gate\nκ$_t$ · e$^{i b_t}$", fc="#fdf1df", ec="#e08a1e", fs=10)
    arrow(ax, (1.55, 2.4), (2.2, 2.4))
    arrow(ax, (4.9, 2.4), (7.1, 5.25), rad=-0.15)
    box(ax, (8.6, 5.1), 3.0, 1.0, "q(x | y, b)\nvon Mises", fs=10)
    arrow(ax, (7.75, 5.6), (8.6, 5.6))
    ax.text(6.0, 0.55, "κ$_t$ → 0:  b is ignored, the posterior comes from neural activity alone",
            ha="center", fontsize=9.5)
    ax.text(6.0, 0.05, "κ$_t$ → ∞:  the posterior is pinned to behavior, x = b", ha="center", fontsize=9.5)
    ax.text(8.6, 3.9, "z = z$_ψ$(y) + κ e$^{ib}$\nposterior mean = arg z\nconcentration = |z|",
            fontsize=9, va="top", color="#555555")
    fig.savefig(f"{OUT}/model_schematic.png")
    plt.close(fig)


def fig_gate():
    x = np.linspace(-np.pi, np.pi, 720)
    zy = 3.0 * np.exp(1j * 0.6)      # neural evidence decoded at 0.6 rad
    b = -0.4                         # measured behavior
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.1))
    ax = axs[0]
    for k, col in zip((0.0, 1.0, 4.0, 20.0), ("#1f5fa8", "#4c9ad4", "#e08a1e", "#d62728")):
        z = zy + k * np.exp(1j * b)
        d = np.exp(np.real(np.conj(z) * np.exp(1j * x)))
        d /= np.trapezoid(d, x)
        ax.plot(x, d, color=col, lw=2, label=f"κ = {k:g}")
    ax.axvline(0.6, color="#1f5fa8", ls=":", lw=1); ax.axvline(b, color="#d62728", ls=":", lw=1)
    ax.text(0.65, ax.get_ylim()[1] * 0.93, "neural\nevidence", fontsize=8, color="#1f5fa8")
    ax.text(b - 0.12, ax.get_ylim()[1] * 0.93, "behavior", fontsize=8, color="#d62728", ha="right")
    ax.set_xlabel("latent angle x (rad)"); ax.set_ylabel("posterior density")
    ax.set_title("Posterior as the gate opens", loc="left"); ax.legend(frameon=False, fontsize=9)
    ax = axs[1]
    ks = np.logspace(-2, 2, 200)
    z = zy + ks * np.exp(1j * b)
    ax.semilogx(ks, np.angle(z), color="#1f5fa8", lw=2, label="posterior mean")
    ax.axhline(0.6, color="#1f5fa8", ls=":", lw=1); ax.axhline(b, color="#d62728", ls=":", lw=1)
    ax.set_xlabel("gate strength κ"); ax.set_ylabel("posterior mean (rad)")
    ax2 = ax.twinx(); ax2.spines["right"].set_visible(True)
    ax2.semilogx(ks, np.abs(z), color="#e08a1e", lw=2, label="concentration")
    ax2.set_ylabel("concentration |z|", color="#e08a1e")
    ax.set_title("Mean moves from neural to behavior", loc="left")
    fig.savefig(f"{OUT}/gate_fusion.png")
    plt.close(fig)



# ---------------------------------------------------------------- smoke tests

def fig_smoke():
    r = json.load(open("results/smoke_results.json"))
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.1), gridspec_kw={"width_ratios": [1, 1.15, 1.15]})
    ax = axs[0]
    lim = [0.05, 15]
    ax.plot(lim, lim, color="#999999", lw=1)
    for key, lab, mk, col in (("strong", "strong neural information (N = 30)", "o", C["amort"]),
                              ("weak", "weak neural information (N = 8)", "^", C["red"])):
        ax.scatter(r[key]["kappa_true"], r[key]["kappa_learned"], marker=mk, s=60, color=col, label=lab, zorder=3)
    ax.scatter([r["noisy"]["kappa_true"]], [r["noisy"]["kappa_learned"]], marker="s", s=60,
               color=C["grid"], label="noisy behavior (N = 30)", zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("true κ"); ax.set_ylabel("learned κ"); ax.set_title("κ recovery", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    for ax, key, ttl in ((axs[1], "strong", "Latent error, strong neural information"),
                         (axs[2], "weak", "Latent error, weak neural information")):
        d = r[key]; kt = d["kappa_true"]
        idx = np.arange(3); w = 0.26
        raw = [d["behavior_mae"][f"cond{c}"] for c in range(3)]
        off = [d["latent_mae_gate_off"][f"cond{c}"] for c in range(3)]
        on = [d["latent_mae_gate_on"][f"cond{c}"] for c in range(3)]
        ax.bar(idx - w, raw, w, color="#bbbbbb", label="raw behavior")
        ax.bar(idx, off, w, color=C["amort2"], label="encoder, gate off")
        ax.bar(idx + w, on, w, color=C["amort"], label="encoder, gate on")
        ax.set_xticks(idx); ax.set_xticklabels([f"κ = {k:g}" for k in kt])
        ax.set_ylabel("mean circular error (rad)"); ax.set_title(ttl, loc="left")
        if key == "strong": ax.legend(frameon=False, fontsize=8)
    fig.savefig(f"{OUT}/smoke_results.png")
    plt.close(fig)


def fig_noisy():
    import jax
    import amortized_eiv as ae
    from amortized_eiv import metrics as M
    d = ae.simulate(seed=1, kappas=(0.1,), T_per=3000)
    params, cfg, stats, feat, _ = ae.fit(d["y"], d["b"], d["cond"], 1, steps=3000, seed=0, verbose=False)
    mu, rho, _ = ae.posterior(params, cfg, feat, d["b"], d["cond"])
    aligned, _ = M.align(np.asarray(mu), d["x"])
    yg, _ = ae.generate(params, cfg, jax.random.PRNGKey(7), 3000)
    yg = np.asarray(yg, dtype=float)
    fig, axs = plt.subplots(1, 4, figsize=(16, 3.9))
    ax = axs[0]
    t = np.arange(300)
    wrap = lambda a: np.where(np.abs(np.diff(a, prepend=a[0])) > np.pi, np.nan, a)
    ax.scatter(t, d["b"][t], s=5, color="#aaaaaa", label="measured behavior")
    ax.plot(t, wrap(d["x"][t]), color=C["truth"], lw=1.2, label="true latent")
    ax.plot(t, wrap(aligned[t]), color=C["red"], lw=1.2, ls="--", label="inferred (aligned)")
    ax.set_ylim(-np.pi, np.pi); ax.set_xlabel("time bin"); ax.set_ylabel("angle (rad)")
    ax.set_title("κ = 0.1: behavior is nearly noise", loc="left")
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3)
    ax = axs[1]
    ax.scatter(d["x"], aligned, s=3, color=C["amort"], alpha=0.5)
    ax.plot([-np.pi, np.pi], [-np.pi, np.pi], color="#999999", lw=1)
    ax.set_xlabel("true latent"); ax.set_ylabel("inferred, aligned")
    ax.set_title(f"Latent recovery, error {M.circ_err(aligned, d['x']).mean():.2f} rad", loc="left")
    ax = axs[2]
    a, b = d["y"].mean(0), yg.mean(0)
    ax.scatter(a, b, s=18, color=C["amort"])
    lim = [0, max(a.max(), b.max()) * 1.05]; ax.plot(lim, lim, color="#999999", lw=1)
    ax.set_xlabel("data mean rate"); ax.set_ylabel("generated mean rate")
    ax.set_title(f"Mean rates, r = {np.corrcoef(a, b)[0, 1]:.3f}", loc="left")
    ax = axs[3]
    iu = np.triu_indices(d["y"].shape[1], 1)
    ca, cb = np.cov(d["y"].T)[iu], np.cov(yg.T)[iu]
    ax.scatter(ca, cb, s=5, color=C["amort"], alpha=0.5)
    lim = [min(ca.min(), cb.min()), max(ca.max(), cb.max())]; ax.plot(lim, lim, color="#999999", lw=1)
    ax.set_xlabel("data covariance"); ax.set_ylabel("generated covariance")
    ax.set_title(f"Neural covariance, r = {np.corrcoef(ca, cb)[0, 1]:.3f}", loc="left")
    fig.savefig(f"{OUT}/noisy_generative.png")
    plt.close(fig)


# -------------------------------------------------------------------- speed 1D

def fig_speed1d():
    sp = json.load(open("results/speed_results.json"))
    cv = json.load(open("results/convergence_results.json"))
    lb = json.load(open("results/lbfgs_results.json"))
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [0.9, 1.35, 1.0]})
    ax = axs[0]
    st = sp["steady_ms_per_step"]
    names = ["amortized\nencoder\nG = 128", "exact grid\nno encoder\nG = 128", "QMC\nM = 1024", "QMC\nM = 128"]
    vals = [st["amortized_encoder_G128"], st["exact_grid_marginal_no_encoder_G128"],
            st["qmc_M1024_fixed_kappa"], st["qmc_M128_fixed_kappa"]]
    bars = ax.bar(range(4), vals, color=[C["amort"], C["grid"], C["qmc"], C["qmc"]])
    for i, v in enumerate(vals): ax.text(i, v + 0.1, f"{v:.1f}", ha="center", fontsize=9)
    ax.set_xticks(range(4)); ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("ms per optimizer step"); ax.set_title("Cost per step", loc="left")
    ax = axs[1]
    for rows, col, lab, mk in ((cv["amortized"], C["amort"], "amortized (Adam)", "o"),
                               (cv["qmc_M1024"], C["qmc"], "QMC, 1024 points (Adam)", "s")):
        ax.plot([r["wall_s"] for r in rows], [r["heldout"] for r in rows], marker=mk, color=col, label=lab, lw=1.8)
    lbx = [lb["compile_s"] + r["wall_s_excl_compile"] for r in lb["rows"] if r["iterations"] >= 10]
    lby = [r["heldout"] for r in lb["rows"] if r["iterations"] >= 10]
    ax.plot(lbx, lby, marker="^", color=C["lbfgs"], lw=1.8, label="QMC, 1024 points (L-BFGS)")
    ax.axhline(cv["qmc_M1024"][-1]["heldout"], color=C["qmc"], ls=":", lw=1)
    ax.text(40, cv["qmc_M1024"][-1]["heldout"] - 0.07, "baseline plateau", fontsize=8, color=C["qmc"], ha="right")
    ax.set_xscale("log"); ax.set_ylim(-58.3, -56.9)
    from matplotlib.ticker import NullFormatter, ScalarFormatter
    ax.set_xticks([3, 5, 10, 20, 50]); ax.xaxis.set_major_formatter(ScalarFormatter()); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("wall time including compilation (s)"); ax.set_ylabel("held-out log p(y | b) per bin")
    ax.set_title("Held-out score against time", loc="left"); ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax = axs[2]
    f = sp["fits"]
    items = [("amortized\nlearned prior", f["amortized_free_kappa_3000"], C["amort"]),
             ("exact grid\nno encoder", f["exact_grid_free_kappa_learned_prior_3000"], C["grid"]),
             ("QMC\nM = 1024", f["qmc_M1024_free_kappa_3000"], C["qmc"]),
             ("QMC\nM = 128", f["qmc_M128_free_kappa_3000"], C["qmc"])]
    vals = [it[1]["wall_s"] for it in items]
    ax.bar(range(4), vals, color=[it[2] for it in items])
    for i, it in enumerate(items):
        ax.text(i, vals[i] + 0.6, f"{vals[i]:.1f} s\nκ = {round(it[1]['kappa'] + 1e-9, 2):.2f}", ha="center", fontsize=8)
    ax.axhline(sp["kappa_selection_wall_s"]["preprint_style_qmc_M1024_grid_cv_extrapolated"], color="#999999", ls="--", lw=1)
    ax.text(3.4, sp["kappa_selection_wall_s"]["preprint_style_qmc_M1024_grid_cv_extrapolated"] + 2,
            "grid over κ with 2 folds\n(18 fits, extrapolated)", ha="right", fontsize=8, color="#666666")
    ax.set_ylim(0, 135); ax.set_xticks(range(4)); ax.set_xticklabels([it[0] for it in items], fontsize=8)
    ax.set_ylabel("wall time of one fit (s)"); ax.set_title("Learning κ in one fit (true κ = 3)", loc="left")
    fig.savefig(f"{OUT}/speed_1d.png")
    plt.close(fig)


# ---------------------------------------------------------------------- 2D torus

def fig_torus2d():
    r = json.load(open("results/torus2d_results.json"))
    runs = r["runs"]
    order = [("amortized_S2_h128", "amortized, S = 2, width 128", C["amort2"], "-"),
             ("amortized_S4_h256", "amortized, S = 4, width 256", C["amort"], "-"),
             ("qmc_M1024", "QMC, 1024 points", C["qmc"], "-"),
             ("qmc_M4096", "QMC, 4096 points", C["qmc"], "--"),
             ("grid_G32", "grid, 32 x 32", C["grid"], "-"),
             ("grid_G64", "grid, 64 x 64", C["grid"], "--")]
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.4), gridspec_kw={"width_ratios": [1.0, 1.25, 1.25, 0.9], "wspace": 0.55})
    ax = axs[0]
    vals = [np.mean([runs[s][k]["ms_per_step"] for s in runs]) for k, *_ in order]
    ax.barh(range(6), vals, color=[o[2] for o in order])
    for i, v in enumerate(vals): ax.text(v + 0.6, i, f"{v:.1f}", va="center", fontsize=8)
    ax.set_yticks(range(6)); ax.set_yticklabels([o[1] for o in order], fontsize=8); ax.invert_yaxis()
    ax.set_xlim(0, 40); ax.set_xlabel("ms per optimizer step (mean of two seeds)"); ax.set_title("Cost per step", loc="left")
    for ax, seed, ttl in ((axs[1], "0", "Held-out score, seed 0 (amortized tuned here)"),
                          (axs[2], "1", "Held-out score, seed 1 (clean check)")):
        for k, lab, col, ls in order:
            h = runs[seed][k]["history"]
            ax.plot([x["train_s"] for x in h], [x["heldout"] for x in h], color=col, ls=ls, marker="o", ms=3, lw=1.6, label=lab)
        best = max(x["heldout"] for k, *_ in order[2:] for x in runs[seed][k]["history"])
        ax.set_xscale("log"); ax.set_ylim(best - 6.0, best + 0.6)
        ax.set_xlabel("training time excluding compilation (s)"); ax.set_ylabel("held-out log p(y | b) per bin")
        ax.set_title(ttl, loc="left", fontsize=10)
    ax = axs[3]
    for i, (k, lab, col, ls) in enumerate(order):
        ks = [runs[s][k]["history"][-1]["kappa"] for s in runs]
        ax.scatter([i] * len(ks), ks, color=col, s=45, zorder=3)
    ax.axhline(4.0, color="#444444", ls="--", lw=1); ax.text(5.4, 4.05, "true κ = 4", ha="right", fontsize=8)
    ax.set_xticks(range(6)); ax.set_xticklabels(["amort\nS=2", "amort\nS=4", "QMC\n1024", "QMC\n4096", "grid\n32", "grid\n64"], fontsize=8)
    ax.set_ylim(2.5, 4.4); ax.set_ylabel("learned κ (two seeds)"); ax.set_title("κ recovery", loc="left")
    h, l = axs[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.07))
    fig.savefig(f"{OUT}/torus2d.png")
    plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1:] or ["model", "gate", "smoke", "noisy", "speed", "torus"]
    if "model" in which: fig_model()
    if "gate" in which: fig_gate()
    if "smoke" in which: fig_smoke()
    if "noisy" in which: fig_noisy()
    if "speed" in which: fig_speed1d()
    if "torus" in which: fig_torus2d()
