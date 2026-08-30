"""
PAPER FIGURE: duty-cycle distribution of a luminosity-limited QSO sample at z~6.

f_duty = (time spent above L_bol threshold up to z_target) / (age of the universe
at z_target), for halos with L_bol(z~6) > 10^46.5 erg/s. The "fine" (sub-step)
estimate is plotted; the per-snapshot "rough" version is quantised to ~1/n_snap
and carries no within-snapshot burst information, so it is only printed.

DATA — note this figure does NOT use the main paper fiducial. It comes from a
SEPARATE ``main_evolution_full_history`` run: a MASS-THRESHOLD selection that
evolves ALL alive halos above log M_halo > 11.5 down to z=6.04 (max_snap=39), so
the f_duty distribution is measured on a COMPLETE QSO sample (not first-born, not
tree_subsample). That run is pinned in ``plotting_paper/fiducial_data.py`` as
``DUTY_CYCLE_FIDUCIAL`` and resolved to ``path_file_duty_cycle_fid`` (token
``massthr11.5``, bestfit inherited from ``PAPER_FIDUCIAL``, no subset tag).
Override any token with ``BAQARO_PAPER_DC_<X>``. This reads the full-catalogue
full-history file (no ``tree_subsample`` weights), so no subsample weighting
applies.

Adapted for the paper from the working duty-cycle figure
(same compute_both / fine-vs-rough logic); the changes are the data source
(paper duty-cycle fiducial via ``fiducial_data``), a stable output basename, and
routing the save to the git-tracked ``figures_paper/`` (PDF by default).

Run::

    BAQARO_SAVE_FIGS=1 python -m baqaro.plotting_paper.plotting_duty_cycle
"""

import h5py
import matplotlib.pyplot as plt
import numpy as np

# Paper duty-cycle fiducial (NOT load_data_to_plot — see fiducial_data.py).
from baqaro.plotting_paper.fiducial_data import (
    load_simulation_metadata_full_history,
    path_file_duty_cycle_fid,
)

from baqaro.plotting_paper import plot_config
from baqaro.plotting_common.plot_config import save_fig, maybe_show


# ==============================================================================
# CONFIG
# ==============================================================================
name_fig = "duty_cycle_qso_z6p5"

PATH_FILE_Z6 = path_file_duty_cycle_fid

Z_TARGET = 6.1
L_BOL_THRESHOLD_ERG_S = 10 ** 46.5      # erg/s
L_SUN_ERG_S = 3.828e33                  # erg/s per L_sun
L_BOL_THRESHOLD_LSUN = L_BOL_THRESHOLD_ERG_S / L_SUN_ERG_S


def _stats(label, f_duty):
    print(f"  [{label}] N={f_duty.size:,}  "
          f"median={np.median(f_duty):.4f}  mean={f_duty.mean():.4f}",
          flush=True)
    for q in [0.05, 0.16, 0.50, 0.84, 0.95]:
        print(f"    {int(q*100):2d}% : {np.quantile(f_duty, q):.4f}", flush=True)
    print(f"    fraction with f_duty > 0.5 : {(f_duty > 0.5).mean():.4f}",
          flush=True)


def compute_both():
    """Compute both rough (per-snapshot) and fine (sub-step) f_duty
    distributions on the *same* halos from the dedicated z=6 full-history file.
    """
    print(f"Loading {PATH_FILE_Z6}", flush=True)
    with h5py.File(PATH_FILE_Z6, "r") as f:
        data = load_simulation_metadata_full_history(f)
        redshifts = data["redshifts"]
        delta_times = data["delta_times_snapshots"]   # Gyr (per-snapshot dt)
        ages = data["ages_of_the_universe"]            # Gyr
        loader = data["loader"]
        full_history_loader = data["full_history_loader"]

        snap_target = int(np.argmin(np.abs(redshifts - Z_TARGET)))
        z_actual = float(redshifts[snap_target])
        age_target = float(ages[snap_target])
        print(f"snap_target={snap_target}, z={z_actual:.3f}, "
              f"age={age_target*1e3:.0f} Myr", flush=True)

        # ---- Identify the QSO sample at z_target via the FINE Lbol ----
        # Use sub-step luminosities (closer to true instantaneous Lbol at z_target).
        times = full_history_loader.times_full_history       # (n_steps,) Gyr
        Lbols_fh = full_history_loader.Lbols_full_history    # (n_steps, n_halos)
        step_target = int(np.argmin(np.abs(times - age_target)))
        print(f"sub-step grid: {times.size} steps; step_target={step_target} "
              f"(t={times[step_target]*1e3:.0f} Myr, dt~{np.median(np.diff(times))*1e3:.2f} Myr)",
              flush=True)

        Lbol_at_target_fh = Lbols_fh[step_target].astype(np.float64)
        sel_idx = np.flatnonzero(Lbol_at_target_fh > L_BOL_THRESHOLD_LSUN)
        n_sel = sel_idx.size
        print(f"\nQSO sample (L_bol > 10^46.5 erg/s at sub-step step_target): "
              f"N={n_sel:,}", flush=True)
        if n_sel == 0:
            return None

        # ---- (B) Fine: integrate sub-step Lbol > threshold ----
        n_steps_used = step_target + 1
        dt_full = np.gradient(times)
        Lbols_earlier = Lbols_fh[:n_steps_used][:, sel_idx].astype(np.float64)
        above_fine = Lbols_earlier > L_BOL_THRESHOLD_LSUN
        time_above_fine = (above_fine * dt_full[:n_steps_used, None]).sum(axis=0)
        f_duty_fine = time_above_fine / age_target

        # ---- Episodic lifetime: duration of the CURRENT continuous above-threshold
        # episode ending at z_target — i.e. how long ago the QSO last crossed above
        # the threshold (and stayed above). The selected halos are above threshold
        # at step_target by construction, so this is the trailing run of consecutive
        # True values in above_fine (summed with the same sub-step dt as f_duty).
        step_idx = np.arange(n_steps_used)[:, None]
        last_below = np.where(above_fine, -1, step_idx).max(axis=0)  # -1 if never below
        in_episode = step_idx > last_below[None, :]                 # trailing True run
        time_episode_fine = (in_episode * dt_full[:n_steps_used, None]).sum(axis=0)
        f_episode_fine = time_episode_fine / age_target

        # ---- (A) Rough: integrate per-snapshot Lbol > threshold (same halos) ----
        time_above_rough = np.zeros(n_sel, dtype=np.float64)
        for i in range(snap_target + 1):
            Lbol_i = loader.get_Lbol(i)[sel_idx]
            time_above_rough[Lbol_i > L_BOL_THRESHOLD_LSUN] += float(delta_times[i])
        f_duty_rough = time_above_rough / age_target

    print("\n" + "=" * 64)
    print("DUTY-CYCLE DISTRIBUTIONS  (same halos, two time resolutions)")
    print("=" * 64)
    _stats("rough/per-snapshot", f_duty_rough)
    _stats("fine/sub-step    ", f_duty_fine)
    _stats("episodic lifetime", f_episode_fine)

    return {
        "f_duty_rough": f_duty_rough,
        "f_duty_fine": f_duty_fine,
        "f_episode_fine": f_episode_fine,
        "z": z_actual,
        "age": age_target,
        "n": n_sel,
    }


def main() -> None:
    """Render the duty-cycle figure, or report that nothing passes the cut."""
    res = compute_both()
    if res is None:
        print("No quasars above threshold -- nothing to plot.")
        return

    # ---- Plot: fine (sub-step) distributions, overlaid in one panel ----
    # Two complementary timescales for the same QSO sample, both as fractions of
    # t_Hubble(z) (Myr on the top axis):
    #   * duty cycle      = total time spent above threshold / t_Hubble   (cumulative)
    #   * episodic lifetime = current continuous above-threshold episode / t_Hubble
    # episode <= duty always (the current burst is part of the total), so the
    # episodic distribution sits at smaller values; their ratio measures how
    # bursty the accretion is. The per-snapshot/rough f_duty is quantised to
    # ~1/n_snap and burst-blind, so it is only printed, not plotted.
    fig, ax = plt.subplots(figsize=(6.0, 5.0), constrained_layout=True)

    bins = np.logspace(-3.5, 0, 36)
    age_target_myr = float(res["age"]) * 1e3  # Gyr -> Myr

    series = [
        ("f_duty_fine", "darkorange", "duty cycle  ($t_{\\rm above}/t_{\\rm H}$)"),
        ("f_episode_fine", "steelblue", "episodic lifetime  ($t_{\\rm episode}/t_{\\rm H}$)"),
    ]
    for key, color, label in series:
        vals = np.asarray(res[key], dtype=float)
        vals_clipped = np.where(vals > 0, vals, bins[0])
        ax.hist(vals_clipped, bins=bins, density=True,
                color=color, edgecolor="black", lw=0.6, alpha=0.55, label=label)
        p16, p50, p84 = np.quantile(vals, [0.16, 0.50, 0.84])
        ax.axvspan(p16, p84, color=color, alpha=0.08, zorder=0)
        ax.axvline(p50, color=color, ls="--", lw=1.6, zorder=5)

    z_show = res["z"]
    n_show = res["n"]

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"fraction of the Hubble time  $t / t_{\rm H}(z)$")
    ax.set_ylabel(r"PDF")
    # Low-x display floor fixed at 6e-4 (below this the histogram is just the
    # clip pile-up at bins[0]).
    ax.set_xlim(6e-4, 1.0)
    ax.legend(loc="upper right", frameon=False, fontsize=11, markerfirst=False)
    ax.minorticks_on()

    # plot_config sets rcParam "xtick.top": True, so the PRIMARY axis also draws
    # its ticks on the top spine. The secondary Myr axis below draws a SECOND set
    # there. The two never coincide — the decades are offset by log10(t_H) (0.96
    # of a decade at z=6.1) — so the top spine ended up with two interleaved tick
    # families at near-but-unequal spacings, which reads as broken ticks. Suppress
    # the primary's top ticks so only the Myr ticks are drawn.
    # direction="out" for the bottom x ticks: the histogram bars run down to the
    # axis and swallow inward-pointing ticks (rcParam default is "in").
    ax.tick_params(axis="x", which="both", top=False, direction="out")

    # Top axis: corresponding time in Myr ( = fraction * t_Hubble(z_target) )
    ax_top = ax.secondary_xaxis(
        "top",
        functions=(lambda f: f * age_target_myr,
                   lambda t: t / age_target_myr),
    )
    ax_top.set_xlabel(r"$t$ [Myr]")
    # NB do NOT call set_xscale("log") or minorticks_on() here: a secondary_xaxis
    # already inherits the parent's log scale AND its log minor locator through
    # the transform. set_xscale is a no-op (verified: identical tick locations);
    # minorticks_on() installs a LINEAR AutoMinorLocator and blows up
    # ("Locator attempting to generate 40184 ticks", MAXTICKS exceeded).

    # Paper convention: save (opt-in via BAQARO_SAVE_FIGS) to figures_paper/ with a
    # stable basename (force_dir pins it there regardless of the data3 toggle),
    # PDF by default; then show unless BAQARO_HEADLESS=1.
    save_fig(fig, plot_config.FIGURES_DIR, name_fig, force_dir=True)
    maybe_show()
    plt.close(fig)


if __name__ == "__main__":
    main()
