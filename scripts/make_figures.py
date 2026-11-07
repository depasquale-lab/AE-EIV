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


# ------------------------------------------------------------------ kernels

KCOL = {"matern": "#222222", "nngp_d1": "#d62728", "nngp_d2": "#e08a1e", "nngp_d3": "#2e9e5b"}
KLAB = {"matern": "Matern-type (default)", "nngp_d1": "ReLU NNGP, depth 1", "nngp_d2": "ReLU NNGP, depth 2",
        "nngp_d3": "ReLU NNGP, depth 3"}
KSPEC = {"matern": ("matern", 1), "nngp_d1": ("nngp", 1), "nngp_d2": ("nngp", 2), "nngp_d3": ("nngp", 3)}


def fig_kernels():
    from amortized_eiv import kernels as K
    J = 12
    taus = {n: K.ring_tau(J, kern, depth) for n, (kern, depth) in KSPEC.items()}
    d = np.linspace(0, np.pi, 300)
    fig = plt.figure(figsize=(15, 7.2))
    gs = fig.add_gridspec(2, 4, hspace=0.5, wspace=0.35)
    ax = fig.add_subplot(gs[0, 0:2])
    for n, t in taus.items():
        k = K.ring_kernel_from_tau(t, d)
        ax.plot(d, k / k[0], color=KCOL[n], lw=2, label=KLAB[n])
    ax.set_xlabel("angular distance between two stimulus values (rad)"); ax.set_ylabel("prior correlation of log rate")
    ax.set_title("Kernel implied by each prior", loc="left"); ax.legend(frameon=False, fontsize=8)
    ax = fig.add_subplot(gs[0, 2:4])
    js = np.arange(1, J + 1)
    for n, t in taus.items():
        ax.semilogy(js, t ** 2, marker="o", ms=4, color=KCOL[n], lw=1.6, label=KLAB[n])
    ax.set_xlabel("harmonic j"); ax.set_ylabel("prior variance of weight at harmonic j")
    ax.set_xticks(js); ax.set_title("Spectrum: depth 1 has no odd harmonics above 1", loc="left")
    rng = np.random.default_rng(3)
    x = np.linspace(-np.pi, np.pi, 400)
    B = np.concatenate([np.cos(x[:, None] * js), np.sin(x[:, None] * js)], axis=1)
    w = rng.standard_normal((2 * J, 4))
    for c, (n, t) in enumerate(taus.items()):
        ax = fig.add_subplot(gs[1, c])
        h = B @ (np.concatenate([t, t])[:, None] * w)
        for q in range(4):
            ax.plot(x, np.exp(h[:, q] - h[:, q].max()), lw=1.4)
        ax.set_title(KLAB[n], loc="left", fontsize=9, color=KCOL[n])
        ax.set_xlabel("stimulus angle (rad)")
        ax.set_ylim(0, 1.05)
        if c == 0: ax.set_ylabel("tuning curve drawn from the prior\n(scaled to peak 1)")
    fig.savefig(f"{OUT}/kernels.png")
    plt.close(fig)


def fig_kernel_results():
    kc = json.load(open("results/kernel_comparison.json"))
    k2 = json.load(open("results/kernel_torus2d.json"))
    t2 = json.load(open("results/torus2d_results.json"))["runs"]
    cases = [("smooth", "kappa8", "smooth\nκ = 8"), ("smooth", "kappa3", "smooth\nκ = 3"),
             ("smooth", "kappa0.5", "smooth\nκ = 0.5"), ("sharp", "kappa3", "sharp bumps\nκ = 3")]
    names = ["nngp_d1", "nngp_d2", "nngp_d3"]
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.2), gridspec_kw={"width_ratios": [1.2, 1.0, 1.0, 1.0], "wspace": 0.4})
    for ax, metric, ttl, ylab in ((axs[0], "heldout", "Held-out score relative to the default prior", "held-out log p(y | b) per bin, difference"),
                                  (axs[1], "tuning_r", "Tuning-curve recovery", "correlation with true curves"),
                                  (axs[2], "latent_err", "Latent error", "mean circular error (rad)")):
        w = 0.2
        for ci, (grp, key, lab) in enumerate(cases):
            base = np.array([r[metric] for r in kc[grp][key]["matern"]])
            if metric == "tuning_r":
                ax.scatter([ci - 1.5 * w] * len(base), base, color=KCOL["matern"], s=26, marker="x", zorder=3)
            if metric == "latent_err":
                ax.scatter([ci - 1.5 * w] * len(base), base, color=KCOL["matern"], s=26, marker="x", zorder=3)
            for ni, n in enumerate(names):
                v = np.array([r[metric] for r in kc[grp][key][n]])
                val = v - base if metric == "heldout" else v
                ax.bar(ci + (ni - 0.5) * w, val.mean(), w, color=KCOL[n])
                ax.scatter([ci + (ni - 0.5) * w] * len(val), val, color="#444444", s=10, zorder=3)
        ax.set_xticks(range(4)); ax.set_xticklabels([c[2] for c in cases], fontsize=8)
        ax.set_title(ttl, loc="left", fontsize=10); ax.set_ylabel(ylab)
        if metric == "heldout": ax.axhline(0, color="#222222", lw=1)
        if metric == "tuning_r": ax.set_ylim(0.6, 1.0)
        if metric == "latent_err": ax.set_ylim(0, None)
    ax = axs[3]
    methods = ["amortized_S4_h256", "qmc_M1024", "grid_G32"]
    mlab = ["amortized\nS=4", "QMC\n1024", "grid\n32 x 32"]
    for mi, mname in enumerate(methods):
        for ni, n in enumerate(["nngp_d1", "nngp_d2"]):
            diffs = [k2[s][n][mname]["heldout"] - t2[s][mname]["history"][-1]["heldout"] for s in ("0", "1")]
            ax.bar(mi + (ni - 0.5) * 0.35, np.mean(diffs), 0.35, color=KCOL[n])
            ax.scatter([mi + (ni - 0.5) * 0.35] * 2, diffs, color="#444444", s=10, zorder=3)
    ax.axhline(0, color="#222222", lw=1); ax.set_xticks(range(3)); ax.set_xticklabels(mlab, fontsize=8)
    ax.set_title("2D torus: held-out relative to default prior", loc="left", fontsize=10)
    ax.set_ylabel("held-out log p(y | b) per bin, difference")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Patch(color=KCOL[n], label=KLAB[n]) for n in names] + [Line2D([], [], color=KCOL["matern"], marker="x", ls="", label="default prior, per seed (panels 2 and 3)"), Line2D([], [], color="#444444", marker="o", ms=4, ls="", label="each seed (bars are means)")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.08))
    fig.savefig(f"{OUT}/kernel_results.png")
    plt.close(fig)


# --------------------------------------------------- neural-encoder-only variant

VCOL = {"gated_encoder": "#1f5fa8", "neural_encoder_only": "#e08a1e", "neural_encoder_only_warp": "#2e9e5b"}
VLAB = {"gated_encoder": "gated encoder", "neural_encoder_only": "neural encoder only",
        "neural_encoder_only_warp": "neural encoder only + learned behavior decoder"}


def fig_variant():
    r = json.load(open("results/neural_encoder_variant.json"))
    names = list(VCOL)
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.3), gridspec_kw={"wspace": 0.32})
    w = 0.26

    def bars(ax, x, key, metric, per_seed=True):
        for ni, n in enumerate(names):
            vals = [x_[metric] for x_ in r[key[0]][key[1]][n]]
            ax.bar(x + (ni - 1) * w, np.mean(vals), w, color=VCOL[n])
            if per_seed: ax.scatter([x + (ni - 1) * w] * len(vals), vals, color="#444444", s=10, zorder=3)

    ax = axs[0]
    ks = ["kappa8", "kappa3", "kappa0.5"]
    for ci, k in enumerate(ks):
        bars(ax, ci, ("strong", k), "behavior_decode_err")
        raw = np.mean([x["raw_behavior_err"] for x in r["strong"][k]["gated_encoder"]])
        ax.scatter([ci], [raw], marker="_", s=500, color="#888888", zorder=2)
    ax.set_xticks(range(3)); ax.set_xticklabels(["κ = 8", "κ = 3", "κ = 0.5"])
    ax.set_ylabel("error decoding behavior from neural activity (rad)")
    ax.set_title("Strong neural information (N = 30)", loc="left")
    ax = axs[1]
    for ci, k in enumerate(["kappa8", "kappa3"]):
        bars(ax, ci, ("weak", k), "kappa")
        ax.hlines(float(k[5:]), ci - 0.45, ci + 0.45, color="#222222", ls="--", lw=1)
    ax.set_xticks(range(2)); ax.set_xticklabels(["true κ = 8", "true κ = 3"]); ax.set_ylabel("learned κ")
    ax.set_title("Weak neural information (N = 8): learned κ", loc="left")
    ax = axs[2]
    bars(ax, 0, ("biased", "kappa6"), "behavior_decode_err")
    raw = np.mean([x["raw_behavior_err"] for x in r["biased"]["kappa6"]["gated_encoder"]])
    ax.scatter([0], [raw], marker="_", s=500, color="#888888", zorder=2)
    for ni, n in enumerate(names):
        k = np.mean([x["kappa"] for x in r["biased"]["kappa6"][n]])
        ax.text((ni - 1) * w, 0.003, f"κ {k:.1f}", ha="center", va="bottom", fontsize=8, color="white")
    ax.set_xticks([0]); ax.set_xticklabels(["b = x + 0.4 + 0.5 sin x, κ = 6"])
    ax.set_xlim(-0.6, 0.6); ax.set_ylabel("error decoding mean behavior from neural activity (rad)")
    ax.set_title("Systematic offset between latent and behavior", loc="left")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Patch(color=VCOL[n], label=VLAB[n]) for n in names] + [
        Line2D([], [], color="#888888", marker="_", ms=14, ls="", label="raw measured behavior (panels 1 and 3)"),
        Line2D([], [], color="#444444", marker="o", ms=4, ls="", label="single seed (bars are means)")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.1))
    fig.savefig(f"{OUT}/neural_encoder_variant.png")
    plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1:] or ["model", "gate", "smoke", "noisy", "speed", "torus"]
    if "model" in which: fig_model()
    if "gate" in which: fig_gate()
    if "smoke" in which: fig_smoke()
    if "noisy" in which: fig_noisy()
    if "speed" in which: fig_speed1d()
    if "torus" in which: fig_torus2d()
    if "kernels" in which: fig_kernels()
    if "kernelresults" in which: fig_kernel_results()
    if "variant" in which: fig_variant()
