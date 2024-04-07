"""
Calculating power loss from partial module shading
==================================================

Example of modeling cell-to-cell mismatch loss from partial module shading.
"""

# %%
# Even though the PV cell is the primary power generation unit, PV modeling is
# often done at the module level for simplicity because module-level parameters
# are much more available and it significantly reduces the computational scope
# of the simulation.  However, module-level simulations are too coarse to be
# able to model effects like cell to cell mismatch or partial shading.  This
# example calculates cell-level IV curves and combines them to reconstruct
# the module-level IV curve.  It uses this approach to find the maximum power
# under various shading and irradiance conditions.
#
# The primary functions used here are:
#
# - :py:meth:`pvlib.pvsystem.calcparams_desoto` to estimate the single
#   diode equation parameters at some specified operating conditions.
# - :py:meth:`pvlib.singlediode.bishop88` to calculate the full cell IV curve,
#   including the reverse bias region.
#
# .. note::
#
#     This example requires the reverse bias functionality added in pvlib 0.7.2
#
# .. warning::
#
#     Modeling partial module shading is complicated and depends significantly
#     on the module's electrical topology.  This example makes some simplifying
#     assumptions that are not generally applicable.  For instance, it assumes
#     that shading only applies to beam irradiance (*i.e.* all cells receive
#     the same amount of diffuse irradiance) and cell temperature is uniform
#     and not affected by cell-level irradiance variation.

from pvlib import pvsystem, singlediode
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from scipy.constants import e as qe, k as kB


# %%
# Simulating a cell IV curve
# --------------------------
#
# First, calculate IV curves for individual cells.  The process is as follows:
#
