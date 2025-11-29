"""
pepq.model
==========

High-level modelling utilities for PepQ.

Public classes
--------------

.. autosummary::
   :toctree: _autosummary

   DockQClassifier
   DockQRegressor
"""

from .classification import DockQClassifier
from .regression import DockQRegressor

__all__ = [
    "DockQClassifier",
    "DockQRegressor",
]
