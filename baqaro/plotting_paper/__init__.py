"""plotting_paper: the publication figure set.

One module per figure. Unlike the exploratory scripts -- which point at
whatever run the ambient ``BAQARO_*`` environment selects -- these pin their
inputs through ``fiducial_data``, so a figure is reproducible from the adopted
fiducial without an environment block, and re-pointing the figure set at a
different run is a single edit in one place.

Figures are written only when ``BAQARO_SAVE_FIGS=1``.
"""
