"""
LAYER 1 - Analytical per-sub-step mean-bias map over (mu, sigma).
================================================================

The cheapest, most fundamental diagnostic: NO sampler, NO halo data. For each
(mu = median log10 eta, sigma = std_0) it computes by quadrature

    exact  = E[eta * (1 - eps(eta))]          (per-sub-step mean Branch A realizes)
    approx = E[eta] * (1 - eps(10**mu))        (per-sub-step mean Branch B realizes)

and the two derived quantities:

    mean_bias  = approx / exact   (= table/exact; < 1 => table UNDER-grows the mean)
    f_eff      = exact / approx   (the f_eff-correction multiplier for mu_scale)

This answers two questions about the derivation in madau_feff.py analytically:
  * WHY under-growth: Madau eps(eta) DECREASES with eta, so (1-eps) is larger in
    the high-eta tail; the exact sum weights that tail, the median-evaluated
    approx does not -> exact > approx -> table under-grows.
  * The (mu, sigma) dependence of the per-sub-step bias (compounded over the run
    by Layer 3).

Outputs: a markdown table to stdout + out/layer1_meanbias_map.png heatmaps.
Run:  python -m baqaro.tests.madau_sampler_bias.test_layer1_efficiency_map
"""
import os
import numpy as np

from . import _common as C

OUTDIR = os.path.join(os.path.dirname(__file__), "out")

# Grids spanning the production landscape. mu = log10(median eta); the ERDF
# fiducial log_eta_mean_0 ~ -1, evolving down the accretion-rate axis, so mu
# ranges roughly [-3, +0.5] across halos/redshift. sigma = std_0 in [0.1, 1.5].
MU_GRID = np.array([-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5])
SIG_GRID = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.3, 1.5])


def compute_maps(rad_eff=0.1):
    nmu, ns = len(MU_GRID), len(SIG_GRID)
    mean_bias = np.zeros((ns, nmu))
    feff = np.zeros((ns, nmu))
    for j, mu in enumerate(MU_GRID):
        for i, s in enumerate(SIG_GRID):
            ex = C.per_substep_growth_mean_exact(mu, s, rad_eff)
            ap = C.per_substep_growth_mean_approx(mu, s, rad_eff)
            mean_bias[i, j] = ap / ex
            feff[i, j] = ex / ap
    return mean_bias, feff


def print_table(mean_bias):
    print("\nLAYER 1: per-sub-step mean-growth bias  (table/exact); <1 = table under-grows")
    print("rows = sigma (std_0), cols = mu (log10 median eta)\n")
    hdr = "sigma\\mu | " + " ".join(f"{mu:+5.1f}" for mu in MU_GRID)
    print(hdr)
    print("-" * len(hdr))
    for i, s in enumerate(SIG_GRID):
        row = " ".join(f"{mean_bias[i, j]:5.3f}" for j in range(len(MU_GRID)))
        print(f"  {s:4.2f}   | {row}")
    print("\n(Per-sub-step. The Layer-3 run compounds this over ~70 snapshots x "
          "n_steps sub-steps.)")


def plot_maps(mean_bias, feff):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot skipped: {e}]")
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ext = [MU_GRID[0], MU_GRID[-1], SIG_GRID[0], SIG_GRID[-1]]
    for ax, data, title, cmap in (
        (axes[0], (1.0 - mean_bias) * 100, "per-sub-step mean deficit [%]\n(100*(1 - table/exact))", "viridis"),
        (axes[1], (feff - 1.0) * 100, "f_eff fix excess [%]\n(100*(exact/approx - 1))", "magma"),
    ):
        im = ax.imshow(data, origin="lower", aspect="auto", extent=ext, cmap=cmap)
        ax.set_xlabel(r"$\mu = \log_{10}\,\eta_{\rm median}$")
        ax.set_ylabel(r"$\sigma$ (std$_0$, dex)")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
        # annotate
        for j, mu in enumerate(MU_GRID):
            for i, s in enumerate(SIG_GRID):
                ax.text(mu, s, f"{data[i, j]:.1f}", ha="center", va="center",
                        fontsize=6, color="w")
    fig.suptitle("Madau-sampler bias: per-sub-step mean (Layer 1, analytical, eps_base=0.1)")
    fig.tight_layout()
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, "layer1_meanbias_map.png")
    fig.savefig(path, dpi=110)
    print(f"\n[saved] {path}")


def main():
    mean_bias, feff = compute_maps()
    print_table(mean_bias)
    plot_maps(mean_bias, feff)
    # headline numbers at the fiducial corner
    print("\nFiducial-ish corner (mu=-1, sigma=0.5): "
          f"per-sub-step mean deficit = {(1 - C.per_substep_growth_mean_approx(-1,0.5)/C.per_substep_growth_mean_exact(-1,0.5))*100:.2f}%")
    print("High-scatter corner   (mu= 0, sigma=1.0): "
          f"per-sub-step mean deficit = {(1 - C.per_substep_growth_mean_approx(0,1.0)/C.per_substep_growth_mean_exact(0,1.0))*100:.2f}%")


if __name__ == "__main__":
    main()
