"""BAQARO — empirical SMBH population modelling in N-body simulations.

Importing any submodule installs the legacy environment-variable aliases (see
:func:`_install_legacy_env_aliases`), so this must stay the first thing that
runs in the package.
"""

import os as _os

#: Canonical prefix for every environment variable this package reads.
ENV_PREFIX = "BAQARO_"

#: Historical prefix, still accepted. The pipeline post-processes halo
#: catalogues from the FLAMINGO simulations, which were *run* with the SWIFT
#: hydrodynamics code -- but nothing here configures or invokes SWIFT, so the
#: old prefix was misleading. It is kept working indefinitely: existing job
#: scripts and shell history should not have to change.
LEGACY_ENV_PREFIX = "SWIFT_"


def _install_legacy_env_aliases():
    """Expose every ``SWIFT_FOO`` variable under the canonical ``BAQARO_FOO``.

    Runs once at import, before any module reads its configuration. The
    canonical name always wins, so a run that sets both is unambiguous, and
    setting neither is unaffected.

    Returns
    -------
    list of str
        The canonical names populated from a legacy variable. Empty in the
        normal case where callers already use ``BAQARO_*``.
    """
    aliased = []
    for key, value in list(_os.environ.items()):
        if not key.startswith(LEGACY_ENV_PREFIX):
            continue
        canonical = ENV_PREFIX + key[len(LEGACY_ENV_PREFIX):]
        if canonical not in _os.environ:
            _os.environ[canonical] = value
            aliased.append(canonical)
    return aliased


LEGACY_ENV_ALIASED = _install_legacy_env_aliases()
