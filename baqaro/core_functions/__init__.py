"""core_functions: the forward model.

Evolves a black-hole population along halo merger trees and writes the
resulting per-snapshot masses and luminosities. Everything else in the package
either feeds this (halo histories, seeding, the ERDF) or consumes its output
(emulation, inference, plotting).

Entry points:

``main_evolution``
    Single-node forward run over a stratified halo subsample.
``main_evolution_chunked``
    Multi-node full-catalogue run, partitioned across chunks and concatenated.
``main_evolution_full_history``
    Sub-step-resolved variant that retains the accretion history within each
    snapshot, for a small selected set of objects.
``halo_mass_histories_saver``
    Preprocesses the raw subhalo catalogue into the mass and accretion-rate
    arrays the drivers read.
"""
