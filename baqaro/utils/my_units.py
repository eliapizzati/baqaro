"""Unit conversion constants shared across the pipeline.

Both FLAMINGO boxes use the same code units, so these are simulation-
independent. If a future simulation changes conventions, add a per-simulation
factor to ``SimSpec`` in :mod:`sim_config` rather than branching here.
"""

#: Code mass unit -> solar masses.
mass_units = 1e7

#: HBT-HERONS mass unit (1e10 Msun) expressed in code mass units.
halo_mass_units_hbt = 1e10 / mass_units
