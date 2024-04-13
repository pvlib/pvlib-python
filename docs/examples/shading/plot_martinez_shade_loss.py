"""
Modelling power loss due to module shading in non-monolithic Si arrays
======================================================================

This example demonstrates how to model power loss due to row-to-row shading in
a PV array comprised of non-monolithic silicon cells.
"""

# %%
# The example is based on the work of Martinez et al. [1]_.
# The model is implemented in :py:func:`pvlib.shading.martinez_shade_loss`.
# This model corrects the beam and circumsolar incident irradiance
# based on the number of shaded *blocks*. A *block* is defined as a
# group of cells that are protected by a bypass diode.
# More information on the *blocks* can be found in the original paper [1]_ and
# in the function documentation.
#
# The following key functions are used in this example:
# 1. :py:func:`pvlib.shading.martinez_shade_loss` to calculate the adjustment
# 2. :py:func:`pvlib.shading.shading_factor1d` to calculate the fraction of
#    shaded surface and the number of shaded *blocks*, due to row-to-row
#    shading
# 
# .. versionadded:: 0.10.5
# .. sectionauthor:: Echedey Luis <echelual@gmail.com>

from pvlib import shading
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# TODO: REBASE FROM SHADING_FACTOR1D

# %%
# yolo
# --------------------------
#

