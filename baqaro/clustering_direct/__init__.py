"""clustering_direct: measure clustering by counting pairs, not by halo model.

The clustering likelihood predicts correlation functions analytically, from the
quasar host-halo mass function convolved with a halo-model bias. That is fast
enough to evaluate inside an MCMC, but it is a model of the clustering rather
than a measurement of it.

This package does the direct alternative: take the 3-D positions from a
forward run, count pairs with Corrfunc, and compare. Agreement between the two
validates the halo-model machinery the fit depends on; disagreement localises
whether a clustering mismatch comes from the model's haloes or from the
analytic step.

Needs the optional ``Corrfunc`` dependency (``pip install '.[clustering]'``).
"""
