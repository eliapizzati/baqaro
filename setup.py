"""Compatibility shim.

All packaging metadata (name, version, dependencies, package discovery)
lives in ``pyproject.toml`` under PEP 621 ``[project]``. This file exists
only so that legacy ``python setup.py`` invocations still work.
"""

from setuptools import setup

setup()
