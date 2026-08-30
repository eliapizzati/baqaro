"""
Overlay the feffcorr vs no-feff forward runs: weighted BHMF + QLF.
=================================================================

Resolves the run family from the usual load_data_to_plot env block (BAQARO_SIM,
BAQARO_MAX_SNAP, BAQARO_BESTFIT_NAME, BAQARO_GROWTH_SUM_MAX, ...), then loads BOTH
the uncorrected (`...hdf5`) and corrected (`..._feffcorr.hdf5`) products and overlays
their weighted BH mass function and quasar luminosity function at a few
redshifts. Interactive (plt.show()).

Run (no BAQARO_HEADLESS so it shows):
  env BAQARO_MAX_SNAP=71 BAQARO_BESTFIT_NAME=qcc_fid1_z0_chunked_noshift_v1 \
      BAQARO_GROWTH_SUM_MAX=4.6 \
      python -m baqaro.tests.madau_sampler_bias.compare_feff_runs
"""
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

from baqaro.plotting_common.load_data_to_plot import path_file, boxsize

LSUN_ERG = 3.828e33          # Lsun -> erg/s
Z_TARGETS = (2.0, 3.0, 4.0)  # redshifts to overlay
V = float(boxsize) ** 3      # Mpc^3 (no h)


def _pair_paths():
    """Return (off_path, on_path) regardless of which the env resolved to."""
    base = path_file[:-len(".hdf5")]
    if base.endswith("_feffcorr"):
        base = base[:-len("_feffcorr")]
    return base + ".hdf5", base + "_feffcorr.hdf5"


def _weighted_lf(values_log, w, bins):
    """Number density dN/dlog/Mpc^3 in the given log-bins."""
    h, _ = np.histogram(values_log, bins=bins, weights=w)
    dlog = bins[1] - bins[0]
    return h / V / dlog


def _snap_indices(redshifts, targets):
    return {zt: int(np.argmin(np.abs(redshifts - zt))) for zt in targets}


def main():
    off_path, on_path = _pair_paths()
    for p, tag in ((off_path, "no-feff"), (on_path, "feffcorr")):
        print(f"{tag:>9}: {os.path.basename(p)}  exists={os.path.exists(p)}")
    if not (os.path.exists(off_path) and os.path.exists(on_path)):
        raise SystemExit("Both files must exist (run the pair first).")

    f_off = h5py.File(off_path, "r")
    f_on = h5py.File(on_path, "r")
    redshifts = np.asarray(f_off["redshifts"])
    w = np.asarray(f_off["subset/weights"]) if "subset" in f_off else None
    snaps = _snap_indices(redshifts, Z_TARGETS)

    m_bins = np.arange(5.0, 11.51, 0.25)      # log10 M_BH [Msun]
    l_bins = np.arange(43.0, 48.01, 0.25)     # log10 L_bol [erg/s]
    m_ctr = 0.5 * (m_bins[:-1] + m_bins[1:])
    l_ctr = 0.5 * (l_bins[:-1] + l_bins[1:])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(Z_TARGETS)))

    for c, zt in zip(colors, Z_TARGETS):
        i = snaps[zt]
        for f, ls, lab in ((f_off, "--", None), (f_on, "-", None)):
            M = np.asarray(f["black_hole_masses_all"][i])
            L = np.asarray(f["Lbols_all"][i])
            born = M > 0
            wb = w[born] if w is not None else np.ones(born.sum())
            bhmf = _weighted_lf(np.log10(M[born]), wb, m_bins)
            emit = L[born] > 0
            qlf = _weighted_lf(np.log10(L[born][emit] * LSUN_ERG), wb[emit], l_bins)
            with np.errstate(divide="ignore"):
                axes[0].plot(m_ctr, np.log10(bhmf), ls, color=c, lw=1.8)
                axes[1].plot(l_ctr, np.log10(qlf), ls, color=c, lw=1.8)
        # legend proxy (one per z)
        axes[0].plot([], [], "-", color=c, label=f"z={zt:.0f}")

    # style + solid/dashed legend
    for ax, xl, ttl in ((axes[0], r"$\log_{10}\,M_{\rm BH}\,[M_\odot]$", "BHMF"),
                        (axes[1], r"$\log_{10}\,L_{\rm bol}\,[{\rm erg\,s^{-1}}]$", "QLF")):
        ax.set_xlabel(xl)
        ax.set_ylabel(r"$\log_{10}\,\phi\;[{\rm Mpc^{-3}\,dex^{-1}}]$")
        ax.set_title(ttl)
        ax.set_ylim(-9, -1)
        ax.grid(alpha=0.25)
    axes[0].plot([], [], "k--", label="no-feff")
    axes[0].plot([], [], "k-", label="feffcorr (fix)")
    axes[0].legend(fontsize=9)
    fig.suptitle("Madau f_eff correction: forward-run BHMF & QLF "
                 "(solid = corrected, dashed = uncorrected)", fontsize=12)
    fig.tight_layout()

    f_off.close(); f_on.close()
    plt.show()


if __name__ == "__main__":
    main()
