"""
Diffuse IAM Calculation
=======================

Integrating an IAM model across angles to determine the overall reflection
loss for diffuse irradiance.
"""

# %%
# The fraction of light reflected from the front of a module depends on the
# angle of incidence (AOI) of the light compared to the panel surface.  The
# steeper the tilt, the larger the reflected fraction is.  The fraction of
# transmitted light to incident light is called the incident angle modifier
# (IAM).  Several models exist to calculate the IAM for a given incidence
# angle (e.g. :py:func:`pvlib.iam.ashrae`, :py:func:`pvlib.iam.martin_ruiz`,
# :py:func:`pvlib.iam.sapm`, :py:func:`pvlib.iam.physical`).
# However, evaluating the IAM for diffuse light is
# not as straightforward because it comes from all directions and therefore
# has a range of angles of incidence.  Here we show how to integrate the effect
# of AOI reflection across this AOI range using the process described in [1]_.
# In particular, we will recreate Figures 3, 4, and 5 in that paper.
#
# References
# ----------
#  .. [1] B. Marion "Numerical method for angle-of-incidence correction
#     factors for diffuse radiation incident photovoltaic modules",
#     Solar Energy, Volume 147, Pages 344-348. 2017.
#     DOI: 10.1016/j.solener.2017.03.027
#
#  .. [2] Duffie, John A. & Beckman, William A. (2013). Solar Engineering
#     of Thermal Processes.  DOI: 10.1002/9781118671603


from pvlib.iam import marion_integrate, physical
import numpy as np
import matplotlib.pyplot as plt


# %%
# IAM Model
# ---------
#
# The IAM model used to generate the figures in [1]_ focuses on the air-glass
# interface.  It uses Snell's, Fresnel's, and Beer's laws to determine the
# amount of light transmitted through the interface as a function of AOI.
# The function :py:func:`pvlib.iam.physical` implements this model, except it
# also includes an exponential term to model attenuation in the glazing layer.
# To be faithful to Marion's implementation, we will disable this extinction
# term by setting the attenuation coefficient ``K`` parameter to zero.
# For more details on this IAM model, see [2]_.
#
# Marion generated correction factor profiles for two cases:  a standard
# uncoated glass with n=1.526 and a glass with anti-reflective (AR) coating
# with n=1.3.  For convenience, we define a helper function for each case.
# Comparing them across AOI recreates Figure 3 in [1]_:


def calc_uncoated(aoi):
    return physical(aoi, n=1.526, K=0)


def calc_ar_coated(aoi):
    return physical(aoi, n=1.3, K=0)


aoi = np.arange(0, 91)
cf_uncoated = calc_uncoated(aoi)
cf_ar_coated = calc_ar_coated(aoi)

plt.plot(aoi, cf_ar_coated, c='b', label='$F_b$, AR coated, n=1.3')
plt.plot(aoi, cf_uncoated, c='r', label='$F_b$, uncoated, n=1.526')
plt.xlabel(r'Angle-of-Incidence, AOI $(\degree)$')
plt.ylabel('Correction Factor')
plt.legend()
plt.ylim([0, 1.2])
plt.grid()

# %%
# Diffuse sky irradiance (:math:`F_{sky}`)
# -----------------------------------------
#
# Now that we have an AOI model, we use :py:func:`pvlib.iam.marion_integrate`
# to integrate it across solid angle and determine diffuse irradiance
# correction factors.  Marion defines three types of diffuse irradiance:
# sky, horizon, and ground-reflected.  The IAM correction factor is evaluated
# independently for each type.
# First we recreate Figure 4 in [1]_, showing the dependence of the sky diffuse
# correction factor on module tilt.

tilts = np.arange(0, 91, 2.5)

iam_uncoated = marion_integrate(calc_uncoated, tilts, 'sky', N=180)
iam_ar_coated = marion_integrate(calc_ar_coated, tilts, 'sky', N=180)

plt.plot(tilts, iam_ar_coated, c='b', marker='^',
         label='$F_{sky}$, AR coated, n=1.3')
plt.plot(tilts, iam_uncoated, c='r', marker='x',
         label='$F_{sky}$, uncoated, n=1.526')
plt.ylim([0.9, 1.0])
plt.xlabel(r'PV Module Tilt, $\beta (\degree)$')
plt.ylabel('Correction Factor')
plt.grid()
plt.legend()
plt.show()


# %%
# Diffuse horizon and ground irradiance (:math:`F_{hor}, F_{grd}`)
# -----------------------------------------------------------------
#
# Now we recreate Figure 5 in [1]_, showing the dependence of the correction
# factors for horizon and ground diffuse irradiance on module tilt.  Note that
# we use 1800 points instead of 180 for the horizon case to match [1]_.

iam_uncoated_grd = marion_integrate(calc_uncoated, tilts, 'ground', N=180)
iam_ar_coated_grd = marion_integrate(calc_ar_coated, tilts, 'ground', N=180)

iam_uncoated_hor = marion_integrate(calc_uncoated, tilts, 'horizon', N=1800)
iam_ar_coated_hor = marion_integrate(calc_ar_coated, tilts, 'horizon', N=1800)

plt.plot(tilts, iam_ar_coated_hor, c='b', marker='^',
         label='$F_{hor}$, AR coated, n=1.3')
plt.plot(tilts, iam_uncoated_hor, c='r', marker='x',
         label='$F_{hor}$, uncoated, n=1.526')
plt.plot(tilts, iam_ar_coated_grd, c='b', marker='s',
         label='$F_{grd}$, AR coated, n=1.3')
plt.plot(tilts, iam_uncoated_grd, c='r', marker='+',
         label='$F_{grd}$, uncoated, n=1.526')
plt.xlabel(r'PV Module Tilt, $\beta (\degree)$')
plt.ylabel('Correction Factor')
plt.grid()
plt.legend()
plt.show()
