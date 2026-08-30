"""Layered investigation of the transfer-table sampler bias under madau+.

Branch B of the accretion engine draws a sum of lognormal Eddington ratios from
a precomputed table and applies the radiative efficiency once, at the median
eta. That is exact for a constant efficiency, but under the madau+ prescription
eps decreases with eta, so the median-evaluated factor under-counts the
high-eta tail and the bright end is under-grown.

The tests here isolate that effect layer by layer -- kernel, mean bias map,
compounding across snapshots, redshift dependence, extreme tail -- so a
discrepancy can be attributed to one of them rather than to the engine as a
whole. ``core_functions/madau_feff.py`` implements the correction they check.
"""
