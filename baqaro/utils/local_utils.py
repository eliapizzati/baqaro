"""Resolution helpers for the N-body boxes.

Canonical paths live in :mod:`my_dir`, canonical units in :mod:`my_units`.
Do not add per-machine paths or duplicate unit constants here.
"""


def get_mass_resolution_simulation(Lbox_Mpc, N_particles_per_side=5040,
                                   Lbox_Mpc0=2800., mass_resolution0=6.72e9,
                                   N_particles_per_side0=5040):
    """Dark-matter particle mass for a given box size and particle count.

    Scaled from a reference simulation, since the particle mass is just the
    mean matter density times the volume per particle:
    ``m_p ~ L^3 / N^3``.

    Parameters
    ----------
    Lbox_Mpc : float
        Box side length of the target simulation, in Mpc/h.
    N_particles_per_side : int
        Cube root of the particle count of the target simulation.
    Lbox_Mpc0, mass_resolution0, N_particles_per_side0 : float
        Reference simulation (default: L2800N5040, m_p = 6.72e9 Msun/h^2).

    Returns
    -------
    float
        Particle mass in the same units as ``mass_resolution0``.
    """
    return (mass_resolution0
            * (Lbox_Mpc / Lbox_Mpc0) ** 3
            * (N_particles_per_side0 / N_particles_per_side) ** 3)
