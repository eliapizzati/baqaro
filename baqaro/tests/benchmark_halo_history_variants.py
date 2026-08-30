#!/usr/bin/env python
"""
Diff two halo_masses_*.npy files produced by the saver under different
configurations (fold on/off, merger_delay_mode old/new).

Reports per-snap totals, n_resolved, HMF at selected redshifts, and a
per-halo final-mass diff distribution. Writes a PNG and a JSON summary.

USAGE
    python benchmark_halo_history_variants.py <base_a> <base_b> \
        [--source machine_igm] [--out-dir <path>]

Where <base_a> / <base_b> are the file *stems* (no .npy, no leading
"halo_masses_" prefix). Example:

    base_a = "L2800N10080_maxsnap71_nboundthresh40_halofilter_global_tdynfraction_0.25_foldmass_instant_new"
    base_b = "L2800N10080_maxsnap144_nboundthresh40_halofilter_global_tdynfraction_0.25_foldmass_instant_new"

NB: the fold-on/fold-off comparison this tool was first written for is no
longer runnable — the legacy token-less (no-fold, instant_old) arrays were
deleted. The tool itself is generic: it diffs any two halo_masses arrays of
identical shape.

The two files must have identical shape (same simulation, same max_snap,
same nbound_threshold, same halo_filtering_mode, same tdyn_fraction).
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from baqaro.utils.my_dir import (
    get_input_path_HBT_data, get_output_path,
)
from baqaro.utils.my_units import mass_units
from baqaro.utils.sim_config import simulation_name


def _halo_dir(source):
    return os.path.join(get_output_path(source=source), "halo_histories")


def _load_masses(base, source):
    path = os.path.join(_halo_dir(source), f"halo_masses_{base}.npy")
    if not os.path.exists(path):
        sys.exit(f"halo_masses file not found: {path}")
    return np.load(path, mmap_mode='r')


def _load_redshifts(n_snap, source):
    z_file = os.path.join(get_input_path_HBT_data(source=source),
                          simulation_name, "output_list.txt")
    return np.loadtxt(z_file)[:n_snap]


def _short(label, n=44):
    if len(label) <= n:
        return label
    return label[:n - 3] + "..."


def diff(base_a, base_b, source="machine_igm", out_dir=None,
         hmf_z_targets=(0.0, 1.0, 3.0, 6.0),
         log_bins=np.linspace(7.0, 15.5, 86)):
    mass_a = _load_masses(base_a, source)
    mass_b = _load_masses(base_b, source)

    if mass_a.shape != mass_b.shape:
        sys.exit(f"shape mismatch: A={mass_a.shape} B={mass_b.shape}")
    n_snap, n_halo = mass_a.shape
    print(f"Comparing two ({n_snap}, {n_halo}) arrays:")
    print(f"  A: {base_a}")
    print(f"  B: {base_b}")

    redshifts = _load_redshifts(n_snap, source)

    # ---- per-snap aggregates (one row at a time; rows are contiguous) ----
    print("Per-snap totals + n_resolved...", flush=True)
    total_a = np.zeros(n_snap, dtype=np.float64)
    total_b = np.zeros(n_snap, dtype=np.float64)
    n_pos_a = np.zeros(n_snap, dtype=np.int64)
    n_pos_b = np.zeros(n_snap, dtype=np.int64)
    for i in range(n_snap):
        ra = mass_a[i]
        rb = mass_b[i]
        total_a[i] = ra.sum(dtype=np.float64)
        total_b[i] = rb.sum(dtype=np.float64)
        n_pos_a[i] = int((ra > 0).sum())
        n_pos_b[i] = int((rb > 0).sum())

    # ---- HMF at selected redshifts ----
    print("HMF at target redshifts...", flush=True)
    snap_targets = [int(np.argmin(np.abs(redshifts - zt)))
                    for zt in hmf_z_targets]
    bin_centers = 0.5 * (log_bins[1:] + log_bins[:-1])
    log_mass_units = np.log10(mass_units)  # convert internal -> log10(Msun)
    hmf_data = {}
    for snap, zt in zip(snap_targets, hmf_z_targets):
        ra = mass_a[snap]
        rb = mass_b[snap]
        log_a = np.log10(ra[ra > 0], dtype=np.float64) + log_mass_units
        log_b = np.log10(rb[rb > 0], dtype=np.float64) + log_mass_units
        h_a, _ = np.histogram(log_a, bins=log_bins)
        h_b, _ = np.histogram(log_b, bins=log_bins)
        hmf_data[zt] = (snap, h_a, h_b)

    # ---- per-halo final-mass diff ----
    print("Per-halo final-snap diff...", flush=True)
    final_snap = n_snap - 1
    fm_a = np.asarray(mass_a[final_snap], dtype=np.float64)
    fm_b = np.asarray(mass_b[final_snap], dtype=np.float64)
    both_pos = (fm_a > 0) & (fm_b > 0)
    rel_diff = (fm_b[both_pos] - fm_a[both_pos]) / fm_a[both_pos]
    log_diff = np.log10(fm_b[both_pos] / fm_a[both_pos])
    a_only = int(((fm_a > 0) & (fm_b == 0)).sum())
    b_only = int(((fm_b > 0) & (fm_a == 0)).sum())

    # ---- output paths ----
    if out_dir is None:
        out_dir = os.path.join(get_output_path(source=source), "..",
                               "plots", "halo_history_benchmark")
    out_dir = os.path.normpath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{base_a}__vs__{base_b}"
    if len(tag) > 200:
        # Fall back to a short hash if the tag is unwieldy.
        import hashlib
        tag = "diff_" + hashlib.md5(tag.encode()).hexdigest()[:12]

    # ---- summary figure ----
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    label_a = _short(base_a)
    label_b = _short(base_b)

    # Panel 1: total mass per snap
    ax = axes[0, 0]
    ax.plot(redshifts, total_a, lw=2, label=f"A: {label_a}")
    ax.plot(redshifts, total_b, lw=2, ls='--', label=f"B: {label_b}")
    ax.set(xlabel='z', ylabel='Σ halo mass (internal units)', yscale='log',
           title='Total halo mass per snap')
    ax.legend(fontsize=7, loc='lower right')
    ax.invert_xaxis()

    # Panel 2: ratio B/A
    ax = axes[0, 1]
    safe_a = np.maximum(total_a, np.finfo(np.float64).tiny)
    ax.plot(redshifts, total_b / safe_a, lw=2)
    ax.axhline(1.0, color='k', ls='--', alpha=0.5)
    ax.set(xlabel='z', ylabel='B / A',
           title='Total-mass ratio per snap')
    ax.invert_xaxis()

    # Panel 3: n_resolved per snap
    ax = axes[0, 2]
    ax.plot(redshifts, n_pos_a, lw=2, label='A')
    ax.plot(redshifts, n_pos_b, lw=2, ls='--', label='B')
    ax.set(xlabel='z', ylabel='# halos with m > 0',
           title='Resolved halos per snap')
    ax.invert_xaxis()
    ax.legend(fontsize=8)

    # Panels 4–5: HMF at the requested redshifts
    cmap = plt.cm.viridis(np.linspace(0.0, 0.85, len(hmf_z_targets)))
    ax_hmf = axes[1, 0]
    ax_ratio = axes[1, 1]
    for color, zt in zip(cmap, hmf_z_targets):
        snap, h_a, h_b = hmf_data[zt]
        ax_hmf.step(bin_centers, h_a, where='mid', color=color,
                    label=f'A z={redshifts[snap]:.2f}', lw=1.4)
        ax_hmf.step(bin_centers, h_b, where='mid', color=color, ls='--',
                    label=f'B z={redshifts[snap]:.2f}', lw=1.4)
        safe_hA = np.maximum(h_a, 1)
        ax_ratio.step(bin_centers, h_b / safe_hA, where='mid', color=color,
                      lw=1.4, label=f'z={redshifts[snap]:.2f}')
    ax_hmf.set(xlabel='log10(M_halo / Msun)', ylabel='# halos',
               yscale='log', title='HMF')
    ax_hmf.legend(fontsize=7, ncol=2)
    ax_ratio.axhline(1.0, color='k', ls='--', alpha=0.5)
    ax_ratio.set(xlabel='log10(M_halo / Msun)', ylabel='HMF_B / HMF_A',
                 title='HMF ratio')
    ax_ratio.legend(fontsize=7)

    # Panel 6: per-halo final-snap diff
    ax = axes[1, 2]
    finite = np.isfinite(log_diff)
    ax.hist(log_diff[finite], bins=200, range=(-1.0, 1.0))
    ax.set(xlabel='log10(M_B / M_A) at final snap (both > 0)',
           ylabel='count', yscale='log',
           title=f'Per-halo final-snap mass diff '
                 f'(A_only={a_only:,}, B_only={b_only:,})')
    ax.axvline(0.0, color='k', ls='--', alpha=0.5)

    fig.suptitle(f"{base_a}\n   vs\n{base_b}", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_png = os.path.join(out_dir, f"summary_{tag}.png")
    fig.savefig(out_png, dpi=120)
    plt.close(fig)

    # ---- JSON summary ----
    ratio = total_b / safe_a
    rel_finite = rel_diff[np.isfinite(rel_diff)]
    summary = {
        'base_a': base_a,
        'base_b': base_b,
        'n_snap': int(n_snap),
        'n_halo': int(n_halo),
        'final_snap_z': float(redshifts[final_snap]),
        'total_mass_ratio': {
            'min': float(np.min(ratio)),
            'max': float(np.max(ratio)),
            'median': float(np.median(ratio)),
            'at_final_snap': float(ratio[final_snap]),
        },
        'n_resolved': {
            'final_snap_A': int(n_pos_a[final_snap]),
            'final_snap_B': int(n_pos_b[final_snap]),
            'max_diff_over_snaps': int(np.max(np.abs(n_pos_b - n_pos_a))),
        },
        'per_halo_final_snap': {
            'A_only_halos': a_only,
            'B_only_halos': b_only,
            'rel_diff_median': float(np.median(rel_finite)) if rel_finite.size else None,
            'rel_diff_p95': float(np.quantile(rel_finite, 0.95)) if rel_finite.size else None,
            'rel_diff_p99': float(np.quantile(rel_finite, 0.99)) if rel_finite.size else None,
            'rel_diff_max': float(np.max(rel_finite)) if rel_finite.size else None,
            'fraction_above_10pct': float(np.mean(rel_finite > 0.1)) if rel_finite.size else None,
        },
        'hmf_at_target_z': {
            f"z={zt}": {
                'snap': int(snap),
                'actual_z': float(redshifts[snap]),
                'A_counts': hmf_data[zt][1].tolist(),
                'B_counts': hmf_data[zt][2].tolist(),
            }
            for zt, (snap, _, _) in [(zt, hmf_data[zt]) for zt in hmf_z_targets]
        },
        'log_bins': log_bins.tolist(),
    }
    out_json = os.path.join(out_dir, f"summary_{tag}.json")
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)

    # ---- console summary ----
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Final-snap z      : {redshifts[final_snap]:.3f}")
    print(f"Total-mass ratio  : median={summary['total_mass_ratio']['median']:.4f}  "
          f"min={summary['total_mass_ratio']['min']:.4f}  "
          f"max={summary['total_mass_ratio']['max']:.4f}")
    print(f"N_resolved diff   : max(|B-A|) over snaps = "
          f"{summary['n_resolved']['max_diff_over_snaps']:,}")
    if rel_finite.size:
        print(f"Per-halo rel-diff : median={summary['per_halo_final_snap']['rel_diff_median']:+.4f}  "
              f"p95={summary['per_halo_final_snap']['rel_diff_p95']:+.4f}  "
              f"p99={summary['per_halo_final_snap']['rel_diff_p99']:+.4f}")
        print(f"  Fraction with M_B > 1.1 * M_A : "
              f"{summary['per_halo_final_snap']['fraction_above_10pct']:.4%}")
    print(f"Outputs           : {out_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('base_a', help="stem of halo_masses_{base_a}.npy")
    parser.add_argument('base_b', help="stem of halo_masses_{base_b}.npy")
    parser.add_argument('--source', default='machine_igm')
    parser.add_argument('--out-dir', default=None)
    args = parser.parse_args()
    diff(args.base_a, args.base_b, source=args.source, out_dir=args.out_dir)


if __name__ == '__main__':
    main()
