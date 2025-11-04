"""
Diffuse Self-Shading
====================

Modeling the reduction in diffuse irradiance caused by row-to-row diffuse
shading.
"""

# %%
# The term "self-shading" usually refers to adjacent rows blocking direct
# irradiance and casting shadows on each other. However, the concept also
# applies to diffuse irradiance because rows block a portion of the sky
# dome even when the sun is high in the sky. The irradiance loss fraction
# depends on how tightly the rows are packed and where on the module the
# loss is evaluated -- a point near the top of edge of a module will see
# more of the sky than a point near the bottom edge.
#
# This example uses the approach presented by Passias and Källbäck in [1]_
# and recreates two figures from that paper using
# :py:func:`pvlib.shading.masking_angle_passias` and
# :py:func:`pvlib.shading.sky_diffuse_passias`.
#
# However, the pvlib-python authors believe that this approach is incorrect.
# A correction is suggested and compared with the diffuse shading as obtained
# with the view factor model.
#
# References
# ----------
#  .. [1] D. Passias and B. Källbäck, "Shading effects in rows of solar cell
#     panels", Solar Cells, Volume 11, Pages 281-291.  1984.
#     DOI: 10.1016/0379-6787(84)90017-6

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

from pvlib import bifacial, shading, irradiance

# %%
# First we'll recreate Figure 4, showing how the average masking angle varies
# with array tilt and array packing. The masking angle of a given point on a
# module is the angle from horizontal to the next row's top edge and represents
# the portion of the sky dome blocked by the next row. Because it changes
# from the bottom to the top of a module, the average across the module is
# calculated. In [1]_, ``k`` refers to the ratio of row pitch to row slant
# height (i.e. 1 / GCR).

surface_tilt = np.arange(0, 90, 0.5)

plt.figure()
for k in [1, 1.5, 2, 2.5, 3, 4, 5, 7, 10]:
    gcr = 1 / k
    psi = shading.masking_angle_passias(surface_tilt, gcr)
    plt.plot(surface_tilt, psi, label=f'k={k}')

plt.xlabel('Inclination angle [degrees]')
plt.ylabel('Average masking angle [degrees]')
plt.legend()
plt.show()

# %%
# So as the array is packed tighter (decreasing ``k``), the average masking
# angle increases.
#
# Next we'll recreate Figure 5. Note that the y-axis here is the ratio of
# diffuse plane of array irradiance (after accounting for shading) to diffuse
# horizontal irradiance. This means that the deviation from 100% is due to the
# combination of self-shading and the fact that being at a tilt blocks off
# the portion of the sky behind the row. Following the approach detailed in
# [1]_, the first effect would be modeled with
# :py:func:`pvlib.shading.sky_diffuse_passias` and the second with
# :py:func:`pvlib.irradiance.isotropic`.

plt.figure()
for k in [1, 1.5, 2, 10]:
    gcr = 1 / k
    psi = shading.masking_angle_passias(surface_tilt, gcr)
    shading_loss = shading.sky_diffuse_passias(psi)
    transposition_ratio = irradiance.isotropic(surface_tilt, dhi=1.0)
    relative_diffuse = transposition_ratio * (1 - shading_loss) * 100  # %
    plt.plot(surface_tilt, relative_diffuse, label=f'k={k}')

plt.xlabel('Inclination angle [degrees]')
plt.ylabel('Relative diffuse irradiance [%]')
plt.ylim(0, 105)
plt.legend()
plt.show()

# %%
# As ``k`` decreases, GCR increases, so self-shading loss increases and
# collected diffuse irradiance decreases.
#
# However, the pvlib-python authors believe that this approach is incorrect.
#
# Instead, the combination of inter-row shading from the previous row and the
# surface tilt blocking the portion of the sky behind the row is obtained by
# applying :py:func:`pvlib.shading.sky_diffuse_passias` on the sum of the
# masking and surface tilt angles (see dashed curves in below figure). The
# difference with the above approach is marginal for a ground coverage ratio
# of 10%, but becomes very significant for high ground coverage ratios.
#
# Alternatively, one can also use :py:func:`bifacial.utils.vf_row_sky_2d_integ`
# (see dotted curve in below figure), with very similar results except for the
# highest ground coverage ratio. It is believed that the deviation is a result
# of an approximation in :py:func:`pvlib.shading.masking_angle_passias` and
# that :py:func:`bifacial.utils.vf_row_sky_2d_integ` provides the most accurate
# result.

color_cycler = cycler('color', ['blue', 'orange', 'green', 'red'])
linestyle_cycler = cycler('linestyle', ['--', ':'])
plt.rc('axes', prop_cycle=color_cycler * linestyle_cycler)
plt.figure()
for k in [1, 1.5, 2, 10]:
    gcr = 1 / k
    psi = shading.masking_angle_passias(surface_tilt, gcr)
    vf1 = (1 - shading.sky_diffuse_passias(surface_tilt + psi)) * 100  # %
    vf2 = bifacial.utils.vf_row_sky_2d_integ(surface_tilt, gcr) * 100  # %
    plt.plot(surface_tilt, vf1, label=f'k={k} passias corrected')
    plt.plot(surface_tilt, vf2, label=f'k={k} vf_row_sky_2d_integ')

plt.xlabel('Inclination angle [degrees]')
plt.ylabel('Relative diffuse irradiance [%]')
plt.ylim(0, 105)
plt.legend()
plt.show()
