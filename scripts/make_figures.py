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

def box(ax, xy, w, h, text, fc="#eef3fa", ec="#1f5fa8", fs=10, lw=1.4, ls="-", tc="black"):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fc, ec=ec, lw=lw, ls=ls))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs, color=tc)


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
    fig, axs = plt.subplots(1, 2, figsize=(15.5, 5.6), gridspec_kw={"width_ratios": [1.0, 1.05], "wspace": 0.12})
    ax = axs[0]
    ax.set_xlim(0, 11); ax.set_ylim(0, 8); ax.axis("off")
    ax.set_title("The model: x = b + noise, then neural activity", loc="left", fontweight="bold")
    node(ax, (1.0, 4.0), 0.62, "b", fc="#e9e9e9")
    node(ax, (5.2, 4.0), 0.62, "x")
    node(ax, (9.8, 4.0), 0.62, "y", fc="#e9e9e9")
    arrow(ax, (1.65, 4.0), (4.55, 4.0))
    ax.text(3.1, 4.55, "+ noise ε", ha="center", fontsize=9.5)
    ax.text(3.1, 3.3, "scale K\n(concentration κ)", ha="center", va="top", fontsize=8.5, color="#555555")
    arrow(ax, (5.85, 4.0), (9.15, 4.0))
    ax.text(7.5, 4.55, "tuning f$_n$(x)", ha="center", fontsize=9.5)
    ax.text(7.5, 3.3, "Poisson spikes", ha="center", va="top", fontsize=8.5, color="#555555")
    ax.text(1.0, 2.9, "measured\nbehavior", ha="center", va="top", fontsize=8.5, color="#555555")
    ax.text(5.2, 3.0, "internal\nrepresentation", ha="center", va="top", fontsize=8.5, color="#555555")
    box(ax, (3.9, 6.3), 2.6, 1.0, "prior p(x)\nlearned from neural data", fc="#eaf6ee", ec="#2e9e5b", fs=8.5)
    arrow(ax, (5.2, 6.25), (5.2, 4.7), color="#2e9e5b")
    ax.text(7.4, 6.45, "takes over when the\nnoise is large", fontsize=8.5, color="#2e9e5b", va="center")
    ax.text(5.5, 1.75, "K = 0:  x = b. This is the conventional supervised tuning-curve model.", ha="center", fontsize=9.5)
    ax.text(5.5, 1.1, "K large:  b is swamped by the noise, and x is set by the neural encoder,", ha="center", fontsize=9.5)
    ax.text(5.5, 0.55, "decoder, and prior. The model is then a neural latent variable model.", ha="center", fontsize=9.5)
    ax = axs[1]
    ax.set_xlim(0, 12); ax.set_ylim(0, 8); ax.axis("off")
    ax.set_title("Relation to the preprint (Garon, Keeley, Williams)", loc="left", fontweight="bold")
    ax.text(5.1, 7.25, "Preprint", ha="center", fontsize=10, fontweight="bold", color="#444444")
    ax.text(9.4, 7.25, "This code", ha="center", fontsize=10, fontweight="bold", color="#1f5fa8")
    rows = [("Model", "x = b + von Mises noise,\nGP tuning, Poisson spikes", "same model"),
            ("Finding x", "sample x: quasi-Monte Carlo\nor particle filter", "encoder q(x | y, b)"),
            ("Noise scale", "grid of κ values, chosen by\ncross-validation per epoch", "learned by gradient,\nper condition"),
            ("Prior on x", "uniform", "learned"),
            ("Without b", "decode with the fitted\ntuning curves", "weight κ set to 0 in the\nencoder, one forward pass")]
    y0 = 6.4
    for i, (lab, a_, b_) in enumerate(rows):
        yy = y0 - i * 1.3
        ax.text(0.2, yy, lab, ha="left", va="center", fontsize=9.5, fontweight="bold")
        box(ax, (2.6, yy - 0.5), 5.0, 1.0, a_, fc="#f3f3f3", ec="#bbbbbb", fs=8.5)
        box(ax, (7.9, yy - 0.5), 3.9, 1.0, b_, fc="#eef3fa", ec="#1f5fa8", fs=8.5)
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
    ax.set_title("Posterior as behavior's weight κ grows", loc="left"); ax.legend(frameon=False, fontsize=9)
    ax = axs[1]
    ks = np.logspace(-2, 2, 200)
    z = zy + ks * np.exp(1j * b)
    ax.semilogx(ks, np.angle(z), color="#1f5fa8", lw=2, label="posterior mean")
    ax.axhline(0.6, color="#1f5fa8", ls=":", lw=1); ax.axhline(b, color="#d62728", ls=":", lw=1)
    ax.set_xlabel("weight of behavior κ (large κ = small noise)"); ax.set_ylabel("posterior mean (rad)")
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
        ax.bar(idx, off, w, color=C["amort2"], label="encoder, neural activity only")
        ax.bar(idx + w, on, w, color=C["amort"], label="encoder with behavior")
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

VCOL = {"gated_encoder": "#1f5fa8", "gated_encoder_warp": "#2e9e5b", "neural_encoder_only": "#e08a1e",
        "neural_encoder_only_warp": "#f2c27a"}
VLAB = {"gated_encoder": "full model (both encoders)", "gated_encoder_warp": "full model + learned behavior decoder",
        "neural_encoder_only": "ablation: behavior encoder removed", "neural_encoder_only_warp": "ablation + learned behavior decoder"}


def fig_variant():
    r = json.load(open("results/neural_encoder_variant.json"))
    names = list(VCOL)
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.3), gridspec_kw={"wspace": 0.32})
    w = 0.2

    def bars(ax, x, key, metric, per_seed=True):
        for ni, n in enumerate(names):
            vals = [x_[metric] for x_ in r[key[0]][key[1]][n]]
            ax.bar(x + (ni - 1.5) * w, np.mean(vals), w, color=VCOL[n])
            if per_seed: ax.scatter([x + (ni - 1.5) * w] * len(vals), vals, color="#444444", s=10, zorder=3)

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
        ax.hlines(float(k[5:]), ci - 0.5, ci + 0.5, color="#222222", ls="--", lw=1)
    ax.set_xticks(range(2)); ax.set_xticklabels(["true κ = 8", "true κ = 3"]); ax.set_ylabel("learned κ")
    ax.set_title("Weak neural information (N = 8): learned κ", loc="left")
    ax = axs[2]
    bars(ax, 0, ("biased", "kappa6"), "behavior_decode_err")
    raw = np.mean([x["raw_behavior_err"] for x in r["biased"]["kappa6"]["gated_encoder"]])
    ax.scatter([0], [raw], marker="_", s=500, color="#888888", zorder=2)
    for ni, n in enumerate(names):
        k = np.mean([x["kappa"] for x in r["biased"]["kappa6"][n]])
        ax.text((ni - 1.5) * w, 0.003, f"κ {k:.1f}", ha="center", va="bottom", fontsize=7, color="white")
    ax.set_xticks([0]); ax.set_xticklabels(["b = x + 0.4 + 0.5 sin x, κ = 6"])
    ax.set_xlim(-0.7, 0.7); ax.set_ylabel("error decoding mean behavior from neural activity (rad)")
    ax.set_title("Systematic offset between latent and behavior", loc="left")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Patch(color=VCOL[n], label=VLAB[n]) for n in names] + [
        Line2D([], [], color="#888888", marker="_", ms=14, ls="", label="raw measured behavior (panels 1 and 3)"),
        Line2D([], [], color="#444444", marker="o", ms=4, ls="", label="single seed (bars are means)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.16))
    fig.savefig(f"{OUT}/neural_encoder_variant.png")
    plt.close(fig)


# ------------------------------------------------------- architecture variants

def fig_architecture():
    ENC, DEC, BEH = ("#eef3fa", "#1f5fa8"), ("#eaf6ee", "#2e9e5b"), ("#fdf1df", "#e08a1e")
    fig, axs = plt.subplots(1, 2, figsize=(16, 5.8))
    for ax in axs:
        ax.set_xlim(0, 12.4); ax.set_ylim(0, 7.6); ax.axis("off")
    ax = axs[0]
    ax.set_title("Full model: two encoders and two decoders", loc="left", fontweight="bold", fontsize=10.5)
    node(ax, (0.6, 5.3), 0.5, "y", fc="#e9e9e9"); node(ax, (0.6, 2.2), 0.5, "b", fc="#e9e9e9")
    box(ax, (1.5, 4.8), 2.4, 1.0, "neural encoder\nz$_ψ$(y)", fc=ENC[0], ec=ENC[1], fs=9)
    box(ax, (1.5, 1.7), 2.4, 1.0, "behavior encoder\nκ cos(b − g(x))", fc=BEH[0], ec=BEH[1], fs=9)
    arrow(ax, (1.1, 5.3), (1.5, 5.3)); arrow(ax, (1.1, 2.2), (1.5, 2.2))
    node(ax, (4.8, 3.75), 0.4, "+", fs=13)
    arrow(ax, (3.9, 5.3), (4.6, 4.05)); arrow(ax, (3.9, 2.2), (4.6, 3.45))
    node(ax, (6.3, 3.75), 0.5, "x")
    arrow(ax, (5.2, 3.75), (5.8, 3.75), "q(x|y,b)", off=(0, 0.45), fs=8)
    box(ax, (7.6, 4.8), 2.5, 1.0, "neural decoder\np(y|x)", fc=DEC[0], ec=DEC[1], fs=9)
    box(ax, (7.6, 1.7), 2.5, 1.0, "behavior decoder\nVM(b | g(x), κ)", fc=DEC[0], ec=DEC[1], fs=9)
    arrow(ax, (6.7, 4.0), (7.6, 5.1)); arrow(ax, (6.7, 3.5), (7.6, 2.4))
    node(ax, (11.2, 5.3), 0.5, "ŷ", fc="#e9e9e9"); node(ax, (11.2, 2.2), 0.5, "b̂", fc="#e9e9e9")
    arrow(ax, (10.1, 5.3), (10.7, 5.3)); arrow(ax, (10.1, 2.2), (10.7, 2.2))
    ax.text(2.7, 1.3, "weight κ: effectively absent when\nthe noise is large (κ → 0)", fontsize=8, color="#c0701a", ha="center", va="top")
    ax.text(6.2, 0.55, "κ → 0 (large noise): the behavior encoder drops out. The neural encoder, neural decoder,\nand behavior decoder remain. g(x) = x by default, or a learned warp.",
            ha="center", va="top", fontsize=9)
    ax = axs[1]
    ax.set_title("Ablation: behavior encoder removed", loc="left", fontweight="bold", fontsize=10.5)
    node(ax, (0.6, 5.3), 0.5, "y", fc="#e9e9e9")
    box(ax, (1.5, 4.8), 2.4, 1.0, "neural encoder\nq(x | y)", fc=ENC[0], ec=ENC[1], fs=9)
    arrow(ax, (1.1, 5.3), (1.5, 5.3))
    box(ax, (1.5, 1.7), 2.4, 1.0, "behavior encoder\n(removed in\nevery bin)", fc="white", ec="#aaaaaa", fs=8, ls="--", tc="#999999")
    node(ax, (6.3, 3.75), 0.5, "x")
    arrow(ax, (3.9, 5.3), (5.85, 4.0))
    box(ax, (7.6, 4.8), 2.5, 1.0, "neural decoder\np(y|x)", fc=DEC[0], ec=DEC[1], fs=9)
    box(ax, (7.6, 1.7), 2.5, 1.0, "behavior decoder\nVM(b | g(x), κ)", fc=DEC[0], ec=DEC[1], fs=9)
    arrow(ax, (6.7, 4.0), (7.6, 5.1)); arrow(ax, (6.7, 3.5), (7.6, 2.4))
    node(ax, (11.2, 5.3), 0.5, "ŷ", fc="#e9e9e9"); node(ax, (11.2, 2.2), 0.5, "b", fc="#e9e9e9")
    arrow(ax, (10.1, 5.3), (10.7, 5.3)); arrow(ax, (10.1, 2.2), (10.7, 2.2))
    ax.text(10.9, 3.35, "target", fontsize=8, color="#555555", ha="center")
    ax.text(6.2, 0.55, "Behavior reaches the latent only through its decoder. This tests what the behavior\nencoder adds, and is not the full model.",
            ha="center", va="top", fontsize=9)
    fig.savefig(f"{OUT}/architecture_variants.png")
    plt.close(fig)


# ---------------------------------------------------------------- scorecard

def fig_scorecard():
    sp = json.load(open("results/speed_results.json"))
    cv = json.load(open("results/convergence_results.json"))
    t2 = json.load(open("results/torus2d_results.json"))["runs"]
    fam = {"amort": C["amort"], "amort2": C["amort2"], "qmc": C["qmc"], "grid": C["grid"]}
    fit = sp["fits"]
    ring_k = [("amort.", fit["amortized_free_kappa_3000"]["kappa"] / 3.0, "amort"),
              ("exact grid", fit["exact_grid_free_kappa_learned_prior_3000"]["kappa"] / 3.0, "grid"),
              ("QMC 1024", fit["qmc_M1024_free_kappa_3000"]["kappa"] / 3.0, "qmc"),
              ("QMC 128", fit["qmc_M128_free_kappa_3000"]["kappa"] / 3.0, "qmc")]
    names2 = [("amortized_S2_h128", "amort\nS=2", "amort2"), ("amortized_S4_h256", "amort\nS=4", "amort"),
              ("qmc_M1024", "QMC\n1024", "qmc"), ("qmc_M4096", "QMC\n4096", "qmc"),
              ("grid_G32", "grid\n32", "grid"), ("grid_G64", "grid\n64", "grid")]
    k2 = {k: np.mean([t2[s][k]["history"][-1]["kappa"] for s in t2]) / 4.0 for k, *_ in names2}
    fig, axs = plt.subplots(2, 2, figsize=(14, 8.2), gridspec_kw={"hspace": 0.5, "wspace": 0.28})
    ax = axs[0, 0]
    labs = [r[0].replace(" ", "\n") for r in ring_k] + [n[1] for n in names2]
    vals = [r[1] for r in ring_k] + [k2[n[0]] for n in names2]
    cols = [fam[r[2]] for r in ring_k] + [fam[n[2]] for n in names2]
    xs = list(range(4)) + list(range(5, 11))
    ax.bar(xs, vals, color=cols); ax.axhline(1.0, color="#222222", ls="--", lw=1)
    ax.set_xticks(xs); ax.set_xticklabels(labs, fontsize=7); ax.set_ylim(0.6, 1.1)
    ax.text(1.5, 1.07, "ring (true κ = 3)", ha="center", fontsize=9); ax.text(7.5, 1.07, "2D torus (true κ = 4)", ha="center", fontsize=9)
    ax.set_ylabel("learned κ divided by true κ"); ax.set_title("κ recovery", loc="left")
    ax = axs[0, 1]
    ring_best = max(h["heldout"] for h in cv["qmc_M1024"])
    diffs = [max(h["heldout"] for h in cv["amortized"]) - ring_best]
    for k, *_ in names2[:2][::-1]:
        d = [t2[s][k]["history"][-1]["heldout"] - max(t2[s][b]["history"][-1]["heldout"] for b in ("qmc_M1024", "qmc_M4096", "grid_G32", "grid_G64")) for s in t2]
        diffs.append(np.mean(d))
    ax.bar([0, 2, 3], diffs, color=[C["amort"], C["amort"], C["amort2"]]); ax.axhline(0, color="#222222", lw=1)
    ax.set_xticks([0, 2, 3]); ax.set_xticklabels(["ring\n(fixed κ)", "2D\nS=4", "2D\nS=2"], fontsize=8)
    for x_, d in zip([0, 2, 3], diffs): ax.text(x_, 0.12 if d < 0 else d + 0.12, f"{d:+.2f}".replace("-", "−"), ha="center", fontsize=8)
    ax.set_ylabel("held-out log p(y | b) per bin,\namortized minus best integrator"); ax.set_title("Fit quality (above 0 favors amortized)", loc="left")
    ax = axs[1, 0]
    items = [("preprint-style\ngrid over κ,\n2 folds", sp["kappa_selection_wall_s"]["preprint_style_qmc_M1024_grid_cv_extrapolated"], "#999999"),
             ("amortized\nlearned κ", fit["amortized_free_kappa_3000"]["wall_s"], C["amort"]),
             ("exact grid\nlearned κ", fit["exact_grid_free_kappa_learned_prior_3000"]["wall_s"], C["grid"]),
             ("QMC 1024\nlearned κ", fit["qmc_M1024_free_kappa_3000"]["wall_s"], C["qmc"]),
             ("QMC 128\nlearned κ", fit["qmc_M128_free_kappa_3000"]["wall_s"], C["qmc"])]
    ax.bar(range(5), [i[1] for i in items], color=[i[2] for i in items])
    for i, it in enumerate(items): ax.text(i, it[1] + 2, f"{it[1]:.0f} s", ha="center", fontsize=8)
    ax.set_xticks(range(5)); ax.set_xticklabels([i[0] for i in items], fontsize=7); ax.set_ylabel("wall time to choose κ (s), ring")
    ax.set_title("Cost of choosing κ", loc="left")
    ax = axs[1, 1]
    st = sp["steady_ms_per_step"]
    ring_ms = [("amort.", st["amortized_encoder_G128"], "amort"), ("exact grid", st["exact_grid_marginal_no_encoder_G128"], "grid"),
               ("QMC 1024", st["qmc_M1024_fixed_kappa"], "qmc"), ("QMC 128", st["qmc_M128_fixed_kappa"], "qmc")]
    ms2 = [np.mean([t2[s][k]["ms_per_step"] for s in t2]) for k, *_ in names2]
    labs = [r[0].replace(" ", "\n") for r in ring_ms] + [n[1] for n in names2]
    vals = [r[1] for r in ring_ms] + ms2
    cols = [fam[r[2]] for r in ring_ms] + [fam[n[2]] for n in names2]
    ax.bar(xs, vals, color=cols); ax.set_xticks(xs); ax.set_xticklabels(labs, fontsize=7)
    ax.set_ylim(0, max(vals) * 1.18); ax.text(1.5, max(vals) * 1.08, "ring", ha="center", fontsize=9); ax.text(7.5, max(vals) * 1.08, "2D torus", ha="center", fontsize=9)
    ax.set_ylabel("ms per optimizer step"); ax.set_title("Cost per step", loc="left")
    fig.savefig(f"{OUT}/scorecard.png")
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
    if "architecture" in which: fig_architecture()
    if "scorecard" in which: fig_scorecard()
